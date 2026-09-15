"""
live_pipeline.py
=================
Stage 4/6: wires the forecaster (Stage 3) and the agent planner + safety
layer (Stage 4/Stage 1-notebook) together into a single controller matching
`baseline_agents.ControllerFn`'s signature, so it can be dropped straight
into `run_comparison()` alongside the static and reactive-legacy baselines.

Design walkthrough: notebooks/04_agentic_planner.ipynb.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from agent_planner import plan_with_retry
from digital_twin import SliceObservation
from schemas import NetworkRules, SliceState


def make_live_agentic_controller(
    llm_call_fn: Callable[[str, str], str],
    window_len: int = 10,
):
    """Build a ControllerFn that:
    1. Waits until at least `window_len` timesteps of history exist (a real
       forecaster needs a window to look back over).
    2. From the raw observation history, computes a simple rolling
       mean/std per slice as the forecast -- notebooks/03_forecaster.ipynb's
       *trained* ProbabilisticLSTM is the intended forecaster in a full run;
       this rolling-statistics version is used here so the comparison
       harness (Stage 6) doesn't need to carry a trained model per slice
       through every twin instantiation. Swap in `forecaster.predict()`
       directly for a closer-to-production run (see notebooks/04, Section 4).
    3. Builds `SliceState` objects and calls `agent_planner.plan_with_retry()`.
    """

    def controller(
        timestep: int,
        observation_history: List[Dict[str, SliceObservation]],
        rules: NetworkRules,
        previous_allocations: Dict[str, float],
    ) -> Dict[str, float]:
        if len(observation_history) < window_len:
            return previous_allocations

        slice_states = []
        for name in previous_allocations:
            recent_demands = [h[name].demand_mbps for h in observation_history[-window_len:]]
            mean = float(np.mean(recent_demands))
            std = float(np.std(recent_demands))
            last_obs = observation_history[-1][name]
            slice_states.append(SliceState(
                name=name,
                current_allocation_mbps=previous_allocations[name],
                observed_latency_ms=last_obs.latency_ms,
                observed_throughput_mbps=last_obs.throughput_mbps,
                forecast_mean_mbps=mean,
                forecast_std_mbps=std,
            ))

        plan = plan_with_retry(timestep, slice_states, rules, previous_allocations, llm_call_fn)
        return plan.allocations

    return controller
