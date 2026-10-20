"""Validate scoped trend claims against all measured values, never an interpolant."""


def validate(observations, claims):
    checked = []
    for claim in claims:
        key, start, end = claim["metric"], claim["start_epoch"], claim["end_epoch"]
        if start >= end:
            raise ValueError("A temporal claim needs two ordered endpoints")
        rows = [(row["epoch"], row["metrics"][key]) for row in observations
                if start <= row["epoch"] <= end and row["metrics"].get(key) is not None]
        rows.sort()
        epochs = [e for e, _ in rows]
        if len(epochs) < 2 or len(set(epochs)) != len(epochs):
            raise ValueError("A temporal claim needs distinct measured observations")
        if epochs[0] != start or epochs[-1] != end or epochs != claim["observed_epochs"]:
            raise ValueError("Temporal scope or observed epoch coverage differs")
        values = [v for _, v in rows]
        if any(type(v) not in (int, float) for v in values):
            raise ValueError("Temporal values must be numeric")
        import math
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite temporal measurement")
        relations = dict(
            constant=all(v == values[0] for v in values),
            variable=any(v != values[0] for v in values),
            increase_endpoints=values[-1] > values[0],
            decrease_endpoints=values[-1] < values[0],
            unchanged_endpoints=values[-1] == values[0],
        )
        if not relations.get(claim["relation"], False):
            raise ValueError("False scoped temporal claim: " + str(claim))
        checked.append(dict(claim, values=values))
    return dict(claims=checked, scope="listed measured epochs only",
                free_text_semantics_verified=False, causality_verified=False)
