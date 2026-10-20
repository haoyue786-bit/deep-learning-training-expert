"""Import reviewed conditional lessons, not raw training archives, into a project store."""
import argparse
import json
from pathlib import Path
from project_experience import ProjectStore, file_evidence


def import_lessons(store, path):
    document=json.loads(Path(path).read_text())
    if document.get('project_id')!=store.project_id:
        raise ValueError('Lesson collection belongs to another project')
    receipt=file_evidence(path)
    results=[]
    with store.locked():
        existing={c['case_id']:c for c in store._read('experience.jsonl')}
    for lesson in document['lessons']:
        for key in ('lesson_id','mechanism','conditions','observation','interpretation',
                    'use_when','do_not_infer','counterexamples','evidence'):
            if key not in lesson:raise ValueError('Missing reviewed lesson field: '+key)
        if not lesson['evidence'] or not lesson['conditions'] or not lesson['do_not_infer']:
            raise ValueError('Lessons need conditions, original evidence and bounded conclusions')
        if len(json.dumps(lesson,ensure_ascii=False))>12000:
            raise ValueError('Distill the lesson; original archives stay behind pointers')
        record=dict(case_id=lesson['lesson_id'],project_id=store.project_id,
            action={'operation':'reviewed historical synthesis','mechanism':lesson['mechanism']},
            result={'lesson':{k:lesson[k] for k in ('observation','interpretation','use_when','do_not_infer')},
                    'counterexamples':lesson['counterexamples']},
            outcome='observation',conditions=dict(lesson['conditions'],mechanism=lesson['mechanism']),
            hypotheses=lesson.get('unresolved',[]),
            evidence=lesson['evidence']+[receipt],cost={'new_training_updates':0,'scientific_model_calls':0},
            evaluation_role=lesson.get('evaluation_role','no_quality_evaluation'),
            tags=sorted(set(lesson.get('tags',[])+[lesson['mechanism'],'curated_lesson'])))
        old=existing.get(record['case_id'])
        if old:
            comparable={k:v for k,v in old.items() if k not in ('body_sha256','recorded_at')}
            comparable['tags']=sorted(comparable.get('tags',[]))
            if comparable==record:
                results.append(old)
                continue
        results.append(store.record(record))
    return [r['case_id'] for r in results]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-context',required=True,type=Path)
    parser.add_argument('--lessons',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(import_lessons(ProjectStore.from_file(args.project_context),args.lessons),indent=2))


if __name__=='__main__':main()
