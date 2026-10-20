"""Separate epoch telemetry, checkpoint/metric work, decisions and finite pools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Cadence:
    telemetry_every: int = 1
    evaluate_every: int = 5
    checkpoint_every: int = 5
    agent_every: int = 10
    automatic_refresh_every: int = 10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Positive integer cadence required: {name}")
        if self.telemetry_every != 1:
            raise ValueError("Every-epoch numerical telemetry is required")
        if self.agent_every % self.evaluate_every or self.agent_every % self.checkpoint_every:
            raise ValueError("Every Agent decision must have a fresh evaluation and checkpoint")
        if self.automatic_refresh_every != self.agent_every:
            raise ValueError("Automatic-negative scientific selection follows the ten-epoch Agent cadence")

    @classmethod
    def from_job(cls, job: dict) -> "Cadence":
        policy = job.get("review_cadence")
        if policy is None:
            # Frozen production jobs keep their registered five-epoch behavior.
            interval = job.get("review_interval", 5)
            return cls(evaluate_every=interval, checkpoint_every=interval,
                       agent_every=interval, automatic_refresh_every=interval)
        if policy.get("version") != "separate_observation_decision_v1":
            raise ValueError("Unknown cadence migration version")
        return cls(**{k: policy[k] for k in asdict(cls())})

    def events(self, epoch: int, *, maximum_epoch: int, emergency: bool = False) -> dict:
        if not 0 < epoch <= maximum_epoch:
            raise ValueError("An executed positive epoch within the run cap required")
        end = epoch == maximum_epoch
        return dict(telemetry=True, evaluate=emergency or end or epoch % self.evaluate_every == 0,
                    checkpoint=emergency or end or epoch % self.checkpoint_every == 0,
                    agent=emergency or end or epoch % self.agent_every == 0,
                    automatic_refresh=not end and not emergency and epoch % self.automatic_refresh_every == 0)

    def policy(self) -> dict:
        return dict(version="separate_observation_decision_v1", **asdict(self))

    def generation(self, epoch: int, *, steps_per_epoch: int, maximum_epoch: int) -> dict:
        """Spread one occurrence/patch to the next decision; never renew on resume."""
        if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
            raise ValueError("Actual optimizer updates per epoch required")
        if not 0 <= epoch < maximum_epoch:
            raise ValueError("Pool generation must precede the final epoch")
        next_epoch = min(maximum_epoch, (epoch // self.agent_every + 1) * self.agent_every)
        return dict(generation_epoch=epoch, expiry_epoch=next_epoch,
                    exposure_horizon_updates=(next_epoch-epoch)*steps_per_epoch,
                    maximum_age_epochs=next_epoch-epoch, occurrence_cap_per_generation=1,
                    dose_policy="one occurrence per selected patch; spread, do not multiply dose")


@dataclass
class FinitePoolCursor:
    generation_id: str
    horizon_updates: int
    cursor: int = 0

    def __post_init__(self):
        if not self.generation_id or self.horizon_updates <= 0 or not 0 <= self.cursor <= self.horizon_updates:
            raise ValueError("Invalid finite automatic-negative generation state")

    def take(self, epoch_steps: int) -> tuple[int, int]:
        if epoch_steps <= 0:
            raise ValueError("Positive optimizer exposure required")
        start = self.cursor
        self.cursor = min(self.horizon_updates, self.cursor+epoch_steps)
        return start, self.cursor

    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def restore(cls, saved: dict, *, generation_id: str, horizon_updates: int) -> "FinitePoolCursor":
        if saved["generation_id"] != generation_id or saved["horizon_updates"] != horizon_updates:
            raise ValueError("Cannot rebind a consumed cursor to a different pool or horizon")
        return cls(**saved)


def numerical_anomalies(metrics: dict) -> list[str]:
    """Missing diagnostics stay missing; nonfinite measured scalars are explicit."""
    problems = []
    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items(): visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value): visit(child, f"{path}[{index}]")
        elif isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value):
            problems.append(path)
    visit(metrics, "")
    return problems
