"""CPU-first small-data adapter around the existing Cashew A3/FADA implementation.

Private imagery/checkpoints stay in the explicitly authorized project. The model,
source CE [3,3,10,3], existing 2K local discriminator and global path are reused.
No country validation label is supplied to training_step.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = ["crop_others", "mixed_trees_grassland", "cashew", "built_up"]


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def _read_image(path):
    import rasterio
    with rasterio.open(path) as ds: values = ds.read().astype(np.float32)
    if values.shape != (64, 128, 128) or not np.isfinite(values).all():
        raise ValueError("Expected finite original 64-channel AEF 128x128 patch")
    return torch.from_numpy(values[None])


class CashewTrialAdapter:
    class_names = CLASS_NAMES
    focus_class = 2
    confusing_class = 1
    available_variants = ["M01:adversarial_path_balance", "M02:class_pixel_priority",
                          "M03:pair_ambiguity_reliability", "M04:source_pair_margin", "M08:source_hard_negative"]

    def __init__(self, config):
        import rasterio
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        if self.device.type != "cpu" and config.get("allow_gpu_trial") is not True:
            raise PermissionError("GPU trial requires explicit project authorization; CPU is the default")
        threads = config.get("cpu_threads", 2)
        if not isinstance(threads, int) or not 1 <= threads <= 8: raise ValueError("Bounded CPU worker count required")
        torch.set_num_threads(threads)
        self.job_file = Path(config["job_file"]).resolve(strict=True)
        self.job = json.loads(self.job_file.read_text())
        if self.job.get("target_year") not in (2019, 2025): raise ValueError("Explicit supported target year required")
        runtime = Path(self.job["runtime_provenance"]["root"]).resolve(strict=True)
        if any(name in sys.modules for name in ("architecture", "fada_expert_v4", "train_tile")):
            raise RuntimeError("Use one isolated process per project runtime; do not mix imported snapshots")
        sys.path.insert(0, str(runtime/"scripts"))
        import train_tile
        import architecture
        import fada_expert_v4
        from fada import FineGrainedDomainDiscriminator
        self.runtime = runtime; self.train_tile = train_tile; self.fada = fada_expert_v4
        self._protected_paths = [self.job_file, Path(self.job["source"]["pretrained"]), runtime/"scripts/architecture.py",
                                 runtime/"scripts/fada_expert_v4.py", runtime/"scripts/train_tile.py"]
        self._source_pretrained_hash = _sha(self.job["source"]["pretrained"])
        if self._source_pretrained_hash != self.job["source"]["pretrained_sha256"]:
            raise ValueError("Source checkpoint no longer matches its registered identity")
        seed = int(config.get("seed", 16)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.gan, self.common = train_tile.configure_runtime(self.job, self.device)
        self.model, self.load_stats = architecture.make_model(self.gan, self.job, self.device)
        self.local_d = FineGrainedDomainDiscriminator(768, 4, (256, 128)).to(self.device)
        self.global_d = fada_expert_v4.GlobalDomainDiscriminator(768).to(self.device)
        self.model_opt = torch.optim.AdamW(architecture.model_parameter_groups(self.model, 2e-6, .01, (.9,.999), 1e-8))
        self.local_opt = torch.optim.AdamW(self.local_d.parameters(), lr=2e-6, weight_decay=.01)
        self.global_opt = torch.optim.AdamW(self.global_d.parameters(), lr=2e-6, weight_decay=.01)
        self.models = [self.model, self.local_d, self.global_d]
        self.optimizers = [self.model_opt, self.local_opt, self.global_opt]
        self.source_images, self.source_labels, source_identity = self._source_patch("source_train")
        self.safety_images, self.safety_labels, safety_identity = self._source_patch("source_valid")
        self.validation_patches, self.dataset = self._validation()
        forbidden = set(self.validation_patches)
        target_names = Path(self.job["target_train_list"]).read_text().splitlines()
        target_path = next((Path(self.job["target_root"])/name for name in target_names
                            if str((Path(self.job["target_root"])/name).resolve()) not in forbidden), None)
        if target_path is None: raise ValueError("No separate unlabelled target training patch available")
        self.target_images = _read_image(target_path).to(self.device)
        self.data_manifest = dict(source_train=source_identity, source_validation=safety_identity,
                                  target_unlabelled=dict(path=str(target_path), sha256=_sha(target_path)),
                                  development=self.dataset, augmentation="none; identical reference/treatment inputs",
                                  preserved_architecture=self.job["architecture"],
                                  source_objective="weighted four-class CE [3,3,10,3], ignore10",
                                  human_supplementary_used=False, validation_labels_enter_training=False)
        self._protected_paths.extend(Path(identity[key]) for identity in (source_identity,safety_identity)
                                     for key in ('image','label'))
        self._protected_paths.append(target_path)
        self._protected_paths.extend(Path(path) for path in self.validation_patches)
        point_file=Path(config['validation_file'])
        self._protected_paths.extend(point_file.with_suffix(suffix) for suffix in self.dataset['component_hashes'])

    def _source_patch(self, key):
        import rasterio
        source = self.job["source"]
        for name in Path(source[key]).read_text().splitlines()[:64]:
            label = self.train_tile.source_label_path(Path(source["source_label_root"]), name)
            with rasterio.open(label) as ds: y = ds.read(1).astype(np.int64)
            if y.shape == (128,128) and np.count_nonzero(y == 1) >= 8 and np.count_nonzero(y == 2) >= 8:
                image = Path(source["source_root"])/name
                x = _read_image(image).to(self.device)
                valid = (x[0] != 0).any(0).cpu().numpy()
                y[~valid] = 10
                if np.count_nonzero(y == 1) < 8 or np.count_nonzero(y == 2) < 8: continue
                if not np.isin(y, [0,1,2,3,10]).all(): raise ValueError("Unexpected source label mapping")
                return x, torch.from_numpy(y[None]).to(self.device), dict(image=str(image), image_sha256=_sha(image),
                    label=str(label), label_sha256=_sha(label), role="source_training" if key == "source_train" else "source_validation")
        raise ValueError(f"No source class-pair support in bounded first64 {key} candidates")

    def _validation(self):
        import fiona
        import rasterio
        point_file = Path(self.config["validation_file"]).resolve(strict=True)
        max_patches = self.config.get("validation_patch_limit", 6)
        max_points = self.config.get("validation_point_limit", 24)
        if not 2 <= max_patches <= 16 or not 2 <= max_points <= 64:
            raise ValueError("Small fixed validation panel required; no country-wide scan in the trial loop")
        components = {p.suffix: _sha(p) for p in point_file.parent.glob(point_file.stem+".*") if p.suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")}
        with Path(self.job["target_domain"]["patch_csv"]).open() as f: patches = list(csv.DictReader(f))
        west = min(float(p['west']) for p in patches); east = max(float(p['east']) for p in patches)
        south = min(float(p['south']) for p in patches); north = max(float(p['north']) for p in patches)
        by_patch = {}
        with fiona.open(point_file) as ds:
            if ds.crs.to_epsg() != 4326: raise ValueError("Explicit WGS84 point-to-raster alignment required")
            for feature in ds:
                props = dict(feature['properties']); x,y = feature['geometry']['coordinates']
                if props.get('conflict', 0) or int(props['year']) != self.job['target_year']: continue
                if not (west <= x < east and south < y <= north): continue
                matches = [p for p in patches if float(p['west']) <= x < float(p['east']) and float(p['south']) < y <= float(p['north'])]
                if not matches: continue
                p = min(matches, key=lambda q:q['patch_id']); path = str(Path(p['tif_path']).resolve())
                by_patch.setdefault(path, []).append(dict(point_id=str(props['point_id']), class_id=int(props['class_id']),
                    longitude=x, latitude=y, label_name=props.get('class_name'), tile_id=p['tile_id'], year=self.job['target_year']))
        # Fix a small spatially identified development panel before model predictions.
        chosen = []; covered = set()
        while by_patch and len(chosen) < max_patches:
            def priority(path):
                labels = {r['class_id'] for r in by_patch[path]}
                return (-len((labels & {1,2})-covered), path)
            path = min(by_patch, key=priority); selected = sorted(by_patch.pop(path), key=lambda r:r['point_id'])
            covered.update(r['class_id'] for r in selected); chosen.append((path, selected))
        if not {1,2} <= covered: raise ValueError("This bounded development panel lacks Cashew/mixed-class support")
        groups = {}; all_rows = []
        for path, selected in chosen:
            if len(all_rows) >= max_points: break
            selected = selected[:max_points-len(all_rows)]
            x = _read_image(path)
            with rasterio.open(path) as ds:
                for row in selected:
                    rr,cc = ds.index(row['longitude'], row['latitude'])
                    if not 0 <= rr < 128 or not 0 <= cc < 128 or not (x[0,:,rr,cc] != 0).any():
                        raise ValueError("Validation point has no valid imagery support")
                    row.update(row=rr, col=cc)
            groups[path] = dict(image=x.to(self.device), points=selected)
            all_rows.extend(selected)
        subset_identity = dict(components=components, point_ids=sorted((r['point_id'],r['class_id']) for r in all_rows))
        identity = hashlib.sha256(json.dumps(subset_identity, sort_keys=True).encode()).hexdigest()
        dataset = dict(dataset_id=self.config['dataset_id'], label_version=self.config['label_version'], sha256=identity,
                       role='development', labels_for_gradient=False, point_file=str(point_file), component_hashes=components,
                       point_count=len(all_rows), patch_count=len(groups), year=self.job['target_year'],
                       sampling='fixed bounded class-coverage/spatial panel before predictions; development only',
                       note='Class1 is the supplied mixed-trees/grassland aggregate; errors do not prove pure forest or wetland identity.')
        return groups, dataset

    def snapshot(self):
        return dict(models=[copy.deepcopy(m.state_dict()) for m in self.models],
                    optimizers=[copy.deepcopy(o.state_dict()) for o in self.optimizers],
                    modes=[{name:part.training for name,part in m.named_modules()} for m in self.models],
                    torch_rng=torch.get_rng_state().clone(),
                    cuda_rng=[s.clone() for s in torch.cuda.get_rng_state_all()] if self.device.type == 'cuda' else [],
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())

    def restore(self, state):
        for model, values, modes in zip(self.models, state['models'], state['modes']):
            model.load_state_dict(values, strict=True)
            for name,part in model.named_modules():part.training=modes[name]
        for optimizer, values in zip(self.optimizers, state['optimizers']):
            optimizer.load_state_dict(copy.deepcopy(values)); optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']: torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])

    def protected_artifacts(self):
        return [{'path':str(p), 'sha256':_sha(p)} for p in self._protected_paths]

    def evaluate_points(self):
        was = self.model.training; self.model.eval(); rows = []
        try:
            with torch.inference_mode():
                for item in self.validation_patches.values():
                    logits,_ = self.gan.forward_fuse(self.model, item['image']); probabilities = logits.softmax(1)[0].cpu()
                    for point in item['points']:
                        row = dict(point); row['probabilities'] = probabilities[:,point['row'],point['col']].tolist(); rows.append(row)
        finally: self.model.train(was)
        return rows

    def source_safety(self):
        was = self.model.training; self.model.eval()
        try:
            with torch.inference_mode():
                logits,_ = self.gan.forward_fuse(self.model,self.safety_images)
                prediction = logits.argmax(1); truth=self.safety_labels; valid=truth!=10
                tp=int(((prediction==2)&(truth==2)&valid).sum());fp=int(((prediction==2)&(truth!=2)&valid).sum());fn=int(((prediction!=2)&(truth==2)&valid).sum())
                return dict(scope='one fixed source-validation patch, not full source validation',
                            cashew_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                            source_loss=float(self.train_tile.four_class_source_loss(logits,truth)),tp=tp,fp=fp,fn=fn)
        finally:self.model.train(was)

    def _knowledge(self, logits, features):
        logits=F.interpolate(logits,size=features.shape[-2:],mode='bilinear',align_corners=False)
        return self.fada.class_knowledge_v4(logits,mode='soft',temperature=1.8,confidence_cap=.9,hard_threshold=.9)[0].detach()

    def training_step(self, module, step):
        from training_runtime import algorithms as algorithms
        from training_runtime.registry import resolve
        # Module resolution precedes updates. The same source/unlabelled images
        # and exact full-state restoration are used for both trial arms.
        function=controls=None
        if module:
            role='source_training' if module['module_id'] in ('M04','M08') else 'unlabelled_training'
            function,controls=resolve(module['module_id'],module['variant'],controls=module['controls'],data_role=role,
                                      authorized_variants=self.available_variants)
        for model in self.models:model.train()
        self.model.eval()
        with torch.no_grad():
            sf=self.gan.stage4_map(self.model,self.source_images);tf=self.gan.stage4_map(self.model,self.target_images)
            sl,_=self.gan.forward_fuse(self.model,self.source_images);tl,_=self.gan.forward_fuse(self.model,self.target_images)
            sk=self._knowledge(sl,sf);tk=self._knowledge(tl,tf)
        self.model.train()
        self.local_opt.zero_grad(set_to_none=True);self.global_opt.zero_grad(set_to_none=True)
        if module and module['module_id'] in ('M02','M03'):
            sw=tw=sr=tr=None
            if module['module_id']=='M02':
                multipliers=controls.get('multipliers',[1.]*sk.shape[1])
                sw,_=algorithms.class_pixel_priority(sk,multipliers);tw,_=algorithms.class_pixel_priority(tk,multipliers)
            else:tr,_=function(tk,class_pair=(2,1),**controls)
            ld=algorithms.conditional_domain_loss(self.local_d(sf.detach()),sk,domain='source',priority=sw,reliability=sr)[0]+algorithms.conditional_domain_loss(self.local_d(tf.detach()),tk,domain='target',priority=tw,reliability=tr)[0]
        else:ld,_=self.fada.local_loss(self.local_d,sf.detach(),tf.detach(),sk,tk)
        gd,_=self.fada.global_loss(self.global_d,sf.detach(),tf.detach())
        (ld+gd).backward();self.local_opt.step();self.global_opt.step()
        self.model_opt.zero_grad(set_to_none=True)
        logits,features=self.gan.forward_fuse(self.model,self.source_images)
        cls=self.train_tile.four_class_source_loss(logits,self.source_labels)
        tf=self.gan.stage4_map(self.model,self.target_images);tl,_=self.gan.forward_fuse(self.model,self.target_images);tk=self._knowledge(tl,tf)
        local=self.fada.local_target_adversarial_loss(self.local_d,tf,tk)
        global_=self.fada.global_target_adversarial_loss(self.global_d,tf)
        adv=local+global_;extra=cls.detach()*0.;effect={'supported':False,'nonzero_effect':False}
        if module:
            mid=module['module_id']
            if mid=='M01':
                adv=function(local,global_,**controls);effect=dict(supported=True,nonzero_effect=bool(float((adv-local-global_).detach().abs())>0))
            elif mid=='M02':
                priority,stats=algorithms.class_pixel_priority(tk,controls.get('multipliers',[1.]*tk.shape[1]));changed=algorithms.conditional_domain_loss(self.local_d(tf),tk,domain='source',priority=priority)[0]
                effect=dict(stats,supported=True,nonzero_effect=bool(float((changed-local).detach().abs())>0));adv=changed+global_
            elif mid=='M03':
                reliability,stats=function(tk,class_pair=(2,1),**controls);changed=algorithms.conditional_domain_loss(self.local_d(tf),tk,domain='source',reliability=reliability)[0]
                effect=dict(stats,supported=True,nonzero_effect=bool(float((changed-local).detach().abs())>0));adv=changed+global_
            elif mid=='M04':
                raw,stats=function(features,self.source_labels,class_pair=(2,1),**controls);extra=module['strength']*raw
                effect=dict(stats,nonzero_effect=bool(float(extra.detach().abs())>0))
            elif mid=='M08':
                raw,stats=function(logits,self.source_labels,focus_class=2,negative_class=1,**controls);extra=module['strength']*raw
                effect=dict(stats,nonzero_effect=bool(float(extra.detach().abs())>0))
        total=cls+.004*adv+extra
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite total loss before backward')
        total.backward()
        gradients=[p.grad for p in self.model.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Nonfinite model gradient before update')
        norm=float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients)))
        self.model_opt.step()
        return dict(step=step+1,source_loss=float(cls.detach()),local_adversarial=float(local.detach()),
                    global_adversarial=float(global_.detach()),extra_loss=float(extra.detach()),total_loss=float(total.detach()),
                    model_gradient_norm=norm,module_effect=effect,source_training_patches=1,target_unlabelled_patches=1,
                    model_updates=1,discriminator_updates_per_branch=1,validation_label_reads_in_loss=0,
                    note='Bounded paired mechanism trial; one D update/step for both arms, not a full production reproduction')

    def interface_smoke(self):
        """Call every advertised kernel on actual data without optimizer updates.

        Leaf copies retain the real model's features/logits. Backpropagation here
        verifies each kernel, not a claim of end-to-end treatment efficacy.
        No target validation imagery or labels are accessed by this check.
        """
        from training_runtime.registry import invoke
        from training_runtime.algorithms import conditional_domain_loss
        from training_runtime.trial_bench import state_digest
        before=copy.deepcopy(self.snapshot());identity=state_digest(before);results={}
        protected=self.protected_artifacts()
        try:
            for model in self.models:model.eval()
            with torch.no_grad():
                logits,features=self.gan.forward_fuse(self.model,self.source_images)
                target_features=self.gan.stage4_map(self.model,self.target_images)
                target_logits,_=self.gan.forward_fuse(self.model,self.target_images)
                q=self._knowledge(target_logits,target_features)
                domain=self.local_d(target_features)
            def call(mid,variant,inputs,controls,role):
                return invoke(mid,variant,inputs=inputs,controls=controls,data_role=role,
                              authorized_variants=self.available_variants)
            local=domain.detach().mean().clone().requires_grad_();global_=local.detach().abs().add(1).requires_grad_()
            balanced=call('M01','adversarial_path_balance',dict(local=local,global_loss=global_),
                          dict(local_scale=.5,global_scale=1.5),'unlabelled_training')
            gradients=torch.autograd.grad(balanced,(local,global_))
            results['M01:adversarial_path_balance']=dict(called=True,finite=bool(torch.isfinite(balanced)),
                                                        gradients=[float(g) for g in gradients],
                                                        neutral_reproduced=bool(torch.allclose(call('M01','adversarial_path_balance',dict(local=local,global_loss=global_),{},'unlabelled_training'),local+global_)))
            for mid,variant,controls in [('M02','class_pixel_priority',dict(multipliers=[1,1,1.5,1])),
                                         ('M03','pair_ambiguity_reliability',dict(strength=.5,ecological_strength=.25))]:
                inputs=dict(knowledge=q)
                if mid=='M03':
                    inputs.update(class_pair=(2,1),ecological_support=torch.full_like(q[:,0],.3),
                                  ecological_confidence=torch.full_like(q[:,0],.5))
                weights,stats=call(mid,variant,inputs,controls,'unlabelled_training')
                output=domain.detach().clone().requires_grad_()
                changed,_=conditional_domain_loss(output,q,domain='source',**{('priority' if mid=='M02' else 'reliability'):weights})
                neutral,_=conditional_domain_loss(output,q,domain='source')
                gradient=torch.autograd.grad(changed,output)[0]
                results[mid+':'+variant]=dict(called=True,finite=bool(torch.isfinite(changed)),
                    gradient_norm=float(gradient.norm()),loss_delta=float(changed-neutral),statistics=stats)
            for mid,variant,values,controls in [
                    ('M04','source_pair_margin',dict(features=features.detach().clone().requires_grad_(),labels=self.source_labels,class_pair=(2,1)),dict(margin=.3)),
                    ('M08','source_hard_negative',dict(logits=logits.detach().clone().requires_grad_(),labels=self.source_labels,focus_class=2,negative_class=1),dict(fraction=.25))]:
                loss,stats=call(mid,variant,values,controls,'source_training')
                leaf=values['features' if mid=='M04' else 'logits']
                gradient=torch.autograd.grad(loss,leaf)[0]
                results[mid+':'+variant]=dict(called=True,finite=bool(torch.isfinite(loss)),loss=float(loss),
                    gradient_norm=float(gradient.norm()),statistics=stats)
        finally:self.restore(before)
        restored=state_digest(self.snapshot())==identity
        unchanged=self.protected_artifacts()==protected
        if not restored or not unchanged:raise RuntimeError('Interface check changed protected state')
        if not all(r['finite'] for r in results.values()):raise FloatingPointError('Nonfinite real-data interface result')
        return dict(status='completed',interfaces=results,all_advertised_called=set(results)==set(self.available_variants),
                    actual_source_and_unlabelled_target_data=True,validation_feedback_evaluations=0,labels_enter_kernels_from_validation=False,optimizer_updates=0,
                    reference_restored=restored,protected_inputs_unchanged=unchanged,model_calls=0,
                    scope='real-data kernel/gradient smoke; source feature leaves, not production efficacy')
