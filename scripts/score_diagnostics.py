"""Separate binary score ranking from the delivered decision on allowed labels.

This is a descriptive calculation. A maximum fitted F1 partition is not a
deployment threshold, calibrated probability, independent test or training cause.
The caller owns label permission, binary class mapping, sample identity and
geographic/temporal interpretation. Missing class support stays explicit.
"""
import argparse
import json
import math
from pathlib import Path


def decision_counts(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError('Prediction and label populations differ')
    if any(type(v) is not bool for v in predictions):
        raise ValueError('Require explicit boolean delivered predictions')
    tp = sum(y and p for y, p in zip(labels, predictions))
    fp = sum(not y and p for y, p in zip(labels, predictions))
    fn = sum(y and not p for y, p in zip(labels, predictions))
    tn = sum(not y and not p for y, p in zip(labels, predictions))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
        F1=2. * tp/(2*tp+fp+fn) if tp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        FPR=fp/(fp+tn) if fp+tn else None,
        positive_retention_identifiable=bool(tp+fn))


def analyze(scores, labels, *, delivered_predictions=None):
    scores, labels = list(scores), list(labels)
    if not scores or len(scores) != len(labels):
        raise ValueError('Require a nonempty matched score/label population')
    if any(type(v) not in (bool, int) or v not in [0, 1] for v in labels):
        raise ValueError('Caller must map the actual class to binary labels explicitly')
    labels = [bool(v) for v in labels]
    scores = [float(v) for v in scores]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError('Require finite probability-like scores in [0,1]')
    total = sum(labels); tp = fp = 0; ap = 0.; best = None
    ordered = sorted(zip(scores, labels), reverse=True); i = 0
    while total and i < len(ordered):
        score = ordered[i][0]; j = i; positives = 0
        while j < len(ordered) and ordered[j][0] == score:
            positives += ordered[j][1]; j += 1
        tp += positives; fp += j-i-positives; fn = total-tp
        ap += positives/total * tp/(tp+fp)
        f1 = 2.*tp/(2*tp+fp+fn)
        if best is None or f1 > best['F1']:
            best = dict(threshold=score, TP=tp, FP=fp, FN=fn,
                        TN=len(labels)-total-fp, F1=f1)
        i = j
    output = dict(samples=len(labels), positive=total, negative=len(labels)-total,
        average_precision=ap if total else None,
        maximum_fitted_F1_partition=best,
        threshold_fitted_and_scored_on_same_points=True,
        independent_test_established_by_this_helper=False,
        higher_threshold_identifies_training_duration=False,
        absent_positive_support=not bool(total), absent_negative_support=total==len(labels))
    if delivered_predictions is not None:
        output['delivered_decision'] = decision_counts(labels, list(delivered_predictions))
    return output


def compare_decisions(reference, current, *, class_ids, population_id):
    """Compare actual decisions on an identical explicitly labelled population.

    Each artifact has an artifact_id and rows of sample_id/label/prediction.
    Sample IDs must already encode any needed year/site/repeated-measure identity.
    The caller owns label permission and artifact provenance. No intersection,
    class remapping, threshold fitting or independence claim is performed here.
    """
    class_ids = list(class_ids)
    if (not population_id or not class_ids or
            any(type(x) not in (int, str) for x in class_ids) or
            len(set(class_ids)) != len(class_ids)):
        raise ValueError('Require explicit unique class IDs and a population identity')
    mapping = {value: i for i, value in enumerate(class_ids)}

    def indexed(artifact):
        if not isinstance(artifact.get('artifact_id'), str) or not artifact['artifact_id']:
            raise ValueError('Require an actual artifact identity')
        rows = {}
        for row in artifact['rows']:
            key = row['sample_id']
            if not isinstance(key, str) or not key or key in rows:
                raise ValueError('Sample identities must be nonempty and unique')
            if any(type(row[k]) not in (int, str) or row[k] not in mapping for k in ['label', 'prediction']):
                raise ValueError('Labels and predictions must use the supplied class IDs')
            rows[key] = row
        if not rows:
            raise ValueError('Require a nonempty measured population')
        return rows

    before, after = indexed(reference), indexed(current)
    if before.keys() != after.keys():
        raise ValueError('Paired population differs; do not silently intersect samples')
    size = len(class_ids)
    matrices = [[[0] * size for _ in range(size)] for _ in range(2)]
    transitions = [[[0] * size for _ in range(size)] for _ in range(size)]
    corrected, newly_wrong, wrong_both_changed = [], [], []
    for key in sorted(before):
        left, right = before[key], after[key]
        if left['label'] != right['label']:
            raise ValueError('Paired sample truth changed')
        truth, old, new = [mapping[x] for x in [left['label'], left['prediction'], right['prediction']]]
        matrices[0][truth][old] += 1
        matrices[1][truth][new] += 1
        transitions[truth][old][new] += 1
        if old != truth and new == truth:
            corrected.append(key)
        elif old == truth and new != truth:
            newly_wrong.append(key)
        elif old != new:
            wrong_both_changed.append(key)
    per_class = []
    for index, label in enumerate(class_ids):
        predictions = [[row['prediction'] == label for _, row in sorted(rows.items())] for rows in [before, after]]
        truth = [row['label'] == label for _, row in sorted(before.items())]
        b, a = [decision_counts(truth, predicted) for predicted in predictions]
        gained = [key for key in sorted(before) if before[key]['label'] == label and before[key]['prediction'] != label and after[key]['prediction'] == label]
        lost = [key for key in sorted(before) if before[key]['label'] == label and before[key]['prediction'] == label and after[key]['prediction'] != label]
        new_fp = [key for key in sorted(before) if before[key]['label'] != label and before[key]['prediction'] != label and after[key]['prediction'] == label]
        removed_fp = [key for key in sorted(before) if before[key]['label'] != label and before[key]['prediction'] == label and after[key]['prediction'] != label]
        per_class.append(dict(class_id=label, true_support=sum(truth), before=b, after=a,
                              newly_detected_true_samples=gained, lost_true_samples=lost,
                              new_false_positive_samples=new_fp, removed_false_positive_samples=removed_fp))
    return dict(population_id=population_id, samples=len(before), class_order=class_ids,
                reference_artifact_id=reference['artifact_id'], current_artifact_id=current['artifact_id'],
                confusion_before=matrices[0], confusion_after=matrices[1],
                transition_counts_true_before_after=transitions, corrected_samples=corrected,
                newly_wrong_samples=newly_wrong, changed_prediction_but_still_wrong_samples=wrong_both_changed,
                prediction_changed_samples=sum(before[k]['prediction'] != after[k]['prediction'] for k in before),
                per_class=per_class, independent_samples_established=False,
                interpretation='Observed paired decisions, not a causal mechanism, significance test or newly fitted decoder.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='JSON scores/labels and optional delivered_predictions, already permitted by the caller')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['scores', 'paired-decisions'], default='scores')
    args = parser.parse_args(); values = json.loads(args.input.read_text())
    if args.mode == 'paired-decisions':
        result = compare_decisions(values['reference'], values['current'],
            class_ids=values['class_ids'], population_id=values['population_id'])
    else:
        result = analyze(values['scores'], values['labels'], delivered_predictions=values.get('delivered_predictions'))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
