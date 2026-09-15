"""
reference_llm.py
=================
Stage 4: a deterministic, rule-based stand-in for a real LLM call.

WHY THIS EXISTS: `src/agent_planner.py`'s `plan_with_retry()` takes any
callable matching the `LLMCallFn` signature -- it does not care whether that
callable is a real API call or not. This module provides a fully offline,
free, deterministic implementation of that signature so the whole pipeline
(twin -> forecaster -> planner -> safety layer -> comparison harness) can be
built, tested, and demonstrated end-to-end without an API key or network
access, exactly matching notebooks/04_agentic_planner.ipynb's live run and
notebooks/06_baseline_comparison.ipynb's three-arm comparison.

THIS IS NOT A REPLACEMENT FOR A REAL LLM. It's a hand-written heuristic
(forecast mean + a risk margin, headroom-adjusted for the twin's fading and
contention derating, bounded by capacity/step-change rules) -- it cannot
generalise, has no reasoning, and its `reasoning` field is a fixed template,
not a genuine explanation. Swap `build_reference_llm()` for
`real_anthropic_llm_call` (below) to run the actual live agentic planner
against a real model.
"""

from __future__ import annotations

import json
import os
from typing import Callable

LLMCallFn = Callable[[str, str], str]


def build_reference_llm(
    urllc_headroom: float = 1.9,
    urllc_risk_k: float = 1.5,
    embb_headroom: float = 1.1,
    embb_risk_k: float = 1.2,
) -> LLMCallFn:
    """Build a reference LLMCallFn with tunable headroom/risk parameters.

    `urllc_headroom` inflates the raw (forecast_mean + k*forecast_std)
    target to compensate for the fact that the twin's fading and contention
    mechanisms mean a slice's *effective* capacity is always somewhat below
    its *nominal* allocation (notebooks/02_digital_twin.ipynb, Sections 2-3)
    -- a naive planner that targets only the raw forecast will under-
    provision URLLC and violate QoS more than a static baseline, which is
    exactly what an early, untuned version of this function did (see
    notebooks/04_agentic_planner.ipynb, Section 5, for that exact failure
    mode and the debugging story behind these defaults).
    """

    def reference_llm(system_prompt: str, user_prompt: str) -> str:
        state = json.loads(
            user_prompt.split("Current network state (JSON):")[1].split("Return the allocation")[0]
        )
        slices = {s["name"]: s for s in state["slices"]}
        rules = state["rules"]
        capacity = rules["total_capacity_mbps"]
        urllc_min = rules["urllc_min_guarantee_mbps"]
        max_step = rules["max_step_change_mbps"]

        cur_urllc = slices["URLLC"]["current_allocation_mbps"]
        raw_urllc_target = (
            slices["URLLC"]["forecast_mean_mbps"] + urllc_risk_k * slices["URLLC"]["forecast_std_mbps"]
        ) * urllc_headroom
        step_clipped_urllc = max(cur_urllc - max_step, min(cur_urllc + max_step, raw_urllc_target))
        final_urllc = max(step_clipped_urllc, urllc_min)

        remaining = capacity - final_urllc
        cur_embb = slices["eMBB"]["current_allocation_mbps"]
        raw_embb_target = (
            slices["eMBB"]["forecast_mean_mbps"] + embb_risk_k * slices["eMBB"]["forecast_std_mbps"]
        ) * embb_headroom
        step_clipped_embb = max(cur_embb - max_step, min(cur_embb + max_step, raw_embb_target))
        final_embb = max(0.0, min(step_clipped_embb, remaining))

        return json.dumps({
            "timestep": state["timestep"],
            "allocations": {"URLLC": final_urllc, "eMBB": final_embb},
            "reasoning": (
                "Targeting forecast demand plus a fading/contention-aware safety margin, "
                "bounded by the max per-step change and total capacity rules."
            ),
            "risk_flag": "medium",
        })

    return reference_llm


def real_anthropic_llm_call(system_prompt: str, user_prompt: str) -> str:
    """A real LLM call, wired to the Anthropic API. Requires:
        pip install anthropic
        export ANTHROPIC_API_KEY="your-key-here"

    Pass this function (not `build_reference_llm()`'s output) as the
    `llm_call_fn` argument to `agent_planner.plan_with_retry()` to run the
    live agentic planner against a real model instead of the offline
    reference implementation above.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "real_anthropic_llm_call requires the anthropic package: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=0.2,  # see notebooks/01_agent_overview.ipynb, Section 10
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
