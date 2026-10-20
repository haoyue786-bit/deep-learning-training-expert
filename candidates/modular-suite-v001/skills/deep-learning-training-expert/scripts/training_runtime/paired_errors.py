"""Compare actual same-population class predictions, not only binary score totals."""
from collections import Counter


def compare(before, after, *, focus_class, classes, maximum_examples=12):
    def keyed(rows):
        result = {}
        for row in rows:
            if row["point_id"] in result:
                raise ValueError("Duplicate point identity")
            if row["class_id"] not in classes or row["predicted_class"] not in classes:
                raise ValueError("Unknown class identity")
            result[row["point_id"]] = row
        return result
    left, right = keyed(before), keyed(after)
    if set(left) != set(right):
        raise ValueError("Paired errors need the same point population")
    transitions = Counter()
    changed = []
    old_correct = new_correct = binary_fp_fixed = semantic_fp_fixed = lost_positive = 0
    for point_id, old in left.items():
        new = right[point_id]
        if old["class_id"] != new["class_id"]:
            raise ValueError("Label change breaks a paired response")
        for key in ("longitude", "latitude", "pixel_col", "pixel_row"):
            if old.get(key) != new.get(key):
                raise ValueError("Point location changed")
        truth, a, b = old["class_id"], old["predicted_class"], new["predicted_class"]
        old_correct += a == truth
        new_correct += b == truth
        if truth != focus_class and a == focus_class and b != focus_class:
            binary_fp_fixed += 1
            semantic_fp_fixed += b == truth
        lost_positive += truth == focus_class and a == focus_class and b != focus_class
        if a != b:
            transitions[(truth, a, b)] += 1
            changed.append({"point_id": point_id, "truth": truth, "before": a, "after": b})
    return dict(samples=len(left), four_class_correct_before=old_correct,
                four_class_correct_after=new_correct, removed_focus_false_positives=binary_fp_fixed,
                removed_false_positives_correctly_reclassified=semantic_fp_fixed,
                newly_missed_focus_positives=lost_positive,
                transitions=[dict(truth=t, before=a, after=b, count=n)
                             for (t, a, b), n in sorted(transitions.items())],
                examples=changed[:maximum_examples],
                additional_changed_points=max(0, len(changed)-maximum_examples),
                interpretation="Focus-class suppression and correct multiclass recovery are distinct.")
