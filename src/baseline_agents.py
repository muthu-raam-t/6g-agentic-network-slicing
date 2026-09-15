"""
baseline_agents.py
==================
Stage 6: the comparison baselines and the generic multi-arm evaluation harness.

Design walkthrough: notebooks/06_baseline_comparison.ipynb.

Two baseline controllers, plus `run_comparison()`, which runs any number of
named controllers against independently-constructed (but identically
configured/seeded) twins and reports the metrics described in
notebooks/00_overview.ipynb, Section 10 (P99 URLLC latency, QoS violation
rate, mean eMBB throughput).

IMPORTANT HONESTY NOTE on `ReactiveLegacyAgent`: this is a lightweight,
observation-only reactive controller written to stand in for the *behaviour*
of the earlier semester's trained PPO agent (reacts only to the latest
observed state, no forecast, no explanation) -- it is NOT that original
trained model, since its weights/checkpoint are not part of this rebuild.
If you still have the original trained PPO policy saved, swap this class
for a thin wrapper that loads and queries it instead; `run_comparison()`
only needs any callable with the same signature, so nothing else changes.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from digital_twin import DigitalTwin, SliceObservation
from schemas import NetworkRules

# A controller is any callable with this signature:
#   (timestep, observation_history, rules, previous_allocations) -> new_allocations
ControllerFn = Callable[[int, List[Dict[str, SliceObservation]], NetworkRules, Dict[str, float]], Dict[str, float]]


class StaticBaselineAgent:
    """The control arm: a fixed allocation, never adapts to anything.
    See notebooks/00_overview.ipynb, Section 3.
    """

    def __init__(self, fixed_allocations: Dict[str, float]):
        self.fixed_allocations = dict(fixed_allocations)

    def decide(
        self,
        timestep: int,
        observation_history: List[Dict[str, SliceObservation]],
        rules: NetworkRules,
        previous_allocations: Dict[str, float],
    ) -> Dict[str, float]:
        return dict(self.fixed_allocations)


class ReactiveLegacyAgent:
    """Reactive, observation-only controller standing in for the earlier
    prototype's PPO agent (see module docstring). Shifts bandwidth toward
    whichever slice had the highest *observed* latency last timestep,
    bounded by the network's max-step-change rule, with no forecast and no
    explanation -- this is precisely the "opaque and reactive" behaviour
    notebooks/00_overview.ipynb, Section 3 argues against.
    """

    def __init__(self, urllc_slice_name: str = "URLLC"):
        self.urllc_slice_name = urllc_slice_name

    def decide(
        self,
        timestep: int,
        observation_history: List[Dict[str, SliceObservation]],
        rules: NetworkRules,
        previous_allocations: Dict[str, float],
    ) -> Dict[str, float]:
        if not observation_history:
            # No observations yet -- start from an equal split.
            n = len(previous_allocations) or 1
            return {name: rules.total_capacity_mbps / n for name in previous_allocations}

        latest = observation_history[-1]
        latencies = {name: obs.latency_ms for name, obs in latest.items()}
        total_latency = sum(latencies.values()) or 1.0

        # Desired split: proportional to each slice's share of total observed latency
        # (the slice hurting the most gets the biggest share) -- a crude, purely
        # reactive heuristic by design.
        desired = {
            name: (latencies[name] / total_latency) * rules.total_capacity_mbps
            for name in latencies
        }

        new_allocations = {}
        for name, prev in previous_allocations.items():
            target = desired.get(name, prev)
            step = np.clip(target - prev, -rules.max_step_change_mbps, rules.max_step_change_mbps)
            new_allocations[name] = prev + step

        new_allocations = self._enforce_hard_constraints(new_allocations, rules)
        return new_allocations

    def _enforce_hard_constraints(self, allocations: Dict[str, float], rules: NetworkRules) -> Dict[str, float]:
        """Clip to non-negative, guarantee URLLC's minimum, then rescale
        everything else proportionally so the total never exceeds capacity.
        Unlike the LLM planner, this agent has no safety layer sitting in
        front of it -- these constraints are baked directly into its own
        decision logic instead.
        """
        allocations = {name: max(0.0, v) for name, v in allocations.items()}

        urllc_min = rules.urllc_min_guarantee_mbps
        if allocations.get(self.urllc_slice_name, 0.0) < urllc_min:
            allocations[self.urllc_slice_name] = urllc_min

        total = sum(allocations.values())
        if total > rules.total_capacity_mbps:
            other_names = [n for n in allocations if n != self.urllc_slice_name]
            other_total = sum(allocations[n] for n in other_names) or 1.0
            remaining = max(rules.total_capacity_mbps - allocations[self.urllc_slice_name], 0.0)
            for n in other_names:
                allocations[n] = remaining * (allocations[n] / other_total)

        return allocations


# ---------------------------------------------------------------------------
# Generic multi-arm comparison harness
# ---------------------------------------------------------------------------

def run_comparison(
    twin_factory: Callable[[], DigitalTwin],
    controllers: Dict[str, ControllerFn],
    rules: NetworkRules,
    num_steps: int,
    urllc_slice_name: str = "URLLC",
    embb_slice_name: str = "eMBB",
    qos_threshold_ms: float = 10.0,
) -> Dict[str, dict]:
    """Run every controller in `controllers` against its own independently
    constructed twin (via `twin_factory()`, called once per controller so
    each gets a fresh, identically-configured instance), for `num_steps`
    timesteps, and return per-controller metrics + raw series.

    Each controller starts from an equal split as its "previous allocation"
    on timestep 0.
    """
    results: Dict[str, dict] = {}

    for name, controller_fn in controllers.items():
        twin = twin_factory()
        n_slices = len(twin.slice_names)
        allocations = {s: rules.total_capacity_mbps / n_slices for s in twin.slice_names}

        observation_history: List[Dict[str, SliceObservation]] = []
        urllc_latencies: List[float] = []
        embb_throughputs: List[float] = []

        for t in range(num_steps):
            allocations = controller_fn(t, observation_history, rules, allocations)
            obs = twin.step(allocations)
            observation_history.append(obs)

            urllc_latencies.append(obs[urllc_slice_name].latency_ms)
            if embb_slice_name in obs:
                embb_throughputs.append(obs[embb_slice_name].throughput_mbps)

        urllc_arr = np.array(urllc_latencies)
        results[name] = {
            "p99_urllc_latency_ms": float(np.percentile(urllc_arr, 99)),
            "mean_urllc_latency_ms": float(urllc_arr.mean()),
            "qos_violation_rate": float((urllc_arr > qos_threshold_ms).mean()),
            "mean_embb_throughput_mbps": float(np.mean(embb_throughputs)) if embb_throughputs else None,
            "urllc_latency_series": urllc_latencies,
            "embb_throughput_series": embb_throughputs,
        }

    return results
