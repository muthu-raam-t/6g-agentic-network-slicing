"""
schemas.py
==========
The data contract shared by `agent_planner.py` and `safety_layer.py`.

This is the ONLY thing that is allowed to cross the boundary from the LLM
into the rest of the system (see notebooks/01_agent_overview.ipynb, Section 4).

No network calls, no I/O — this module is pure data definitions so it can be
imported anywhere (planner, safety layer, tests, dashboard) without pulling
in API clients or the twin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class RiskFlag(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class NetworkRules:
    """Hard, non-negotiable operational constraints for one control cycle.

    These come from the network operator, not from the model, and are the
    same object passed to both the prompt (as context) and the safety layer
    (as ground truth for validation) -- see 01_agent_overview.ipynb, Section 2.
    """
    total_capacity_mbps: float
    urllc_min_guarantee_mbps: float
    max_step_change_mbps: float


@dataclass(frozen=True)
class SliceState:
    """Observed state for a single slice at the end of the previous timestep."""
    name: str
    current_allocation_mbps: float
    observed_latency_ms: float
    observed_throughput_mbps: float
    forecast_mean_mbps: float
    forecast_std_mbps: float


@dataclass(frozen=True)
class AllocationPlan:
    """The structured output the LLM planner must return, and the only thing
    the safety layer or the actuation stage ever reads.

    Field meanings are documented in notebooks/01_agent_overview.ipynb, Section 4.
    """
    timestep: int
    allocations: Dict[str, float]
    reasoning: str
    risk_flag: RiskFlag = RiskFlag.MEDIUM

    @staticmethod
    def from_dict(d: dict) -> "AllocationPlan":
        """Construct from a raw (already-JSON-decoded) dict. Raises KeyError/ValueError
        on malformed input -- callers should catch and treat as a Stage-A safety failure.
        """
        return AllocationPlan(
            timestep=int(d["timestep"]),
            allocations={str(k): float(v) for k, v in d["allocations"].items()},
            reasoning=str(d["reasoning"]),
            risk_flag=RiskFlag(d.get("risk_flag", "medium")),
        )

    def to_dict(self) -> dict:
        return {
            "timestep": self.timestep,
            "allocations": dict(self.allocations),
            "reasoning": self.reasoning,
            "risk_flag": self.risk_flag.value,
        }


@dataclass
class ValidationResult:
    """Returned by both safety_layer checks. `ok=False` always carries a
    human-readable `error`, which is exactly the string fed back to the LLM
    on retry (01_agent_overview.ipynb, Section 6).
    """
    ok: bool
    error: str = ""
    stage: str = ""  # "schema" or "constraints"
