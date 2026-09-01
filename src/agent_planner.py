"""
agent_planner.py
================
The live agentic planner: builds the prompt, calls the LLM, and runs the
bounded retry loop against the safety layer.

Design walkthrough: notebooks/01_agent_overview.ipynb, Sections 2, 3, 6.

This module is structured to run stage-by-stage with the rest of the repo:
- Stage 0/1 (now): this file is import-able and its prompt/parsing logic is
  fully testable WITHOUT an API key, via the `llm_call_fn` injection point.
- Stage 4 (`04_agentic_planner.ipynb`): a real client (e.g. the Anthropic or
  OpenAI SDK) is wired in as `llm_call_fn` and run against the live twin.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from schemas import AllocationPlan, NetworkRules, SliceState
from safety_layer import fallback_policy, run_safety_checks

MAX_RETRIES = 2
MODEL_TEMPERATURE = 0.2  # low but non-zero -- see 01_agent_overview.ipynb, Section 10

# Type of the injectable LLM call: (system_prompt, user_prompt) -> raw text response.
# Kept as a plain callable so tests can pass a fake, and Stage 4 can pass a real
# Anthropic/OpenAI client call without this module needing to import either SDK.
LLMCallFn = Callable[[str, str], str]


SYSTEM_PROMPT = """\
You are the automated allocation controller for a 6G radio access network core.
You manage bandwidth for exactly two slices: URLLC (safety-critical, low
latency) and eMBB (best-effort broadband). Every control cycle you receive
the current allocations, the latest observed latency/throughput, a traffic
forecast (mean and standard deviation) for the next cycle, and the network's
hard operational rules.

You must respond with ONLY a single JSON object -- no prose before or after
it, no markdown code fences. The object must have exactly these fields:
{
  "timestep": <int>,
  "allocations": {"URLLC": <mbps float>, "eMBB": <mbps float>},
  "reasoning": "<one or two sentences explaining the decision>",
  "risk_flag": "low" | "medium" | "high"
}

Hard rules you must never violate:
- allocations must sum to no more than total capacity
- URLLC must always receive at least its minimum guaranteed bandwidth
- no allocation may change from the previous timestep by more than the
  maximum allowed step change

Example (illustrative only, use the real numbers you are given at call time):
Input state: URLLC forecast mean=35, std=4; eMBB forecast mean=55, std=10;
capacity=100, URLLC min=30, max step change=20; previous allocations
URLLC=32, eMBB=68.
Output:
{"timestep": 41, "allocations": {"URLLC": 40.0, "eMBB": 60.0},
 "reasoning": "URLLC forecast is rising with moderate uncertainty; shifting
 8 Mbps from eMBB keeps headroom above the forecast mean without breaching
 the per-step change limit.", "risk_flag": "medium"}
"""


def build_state_context(
    timestep: int,
    slice_states: List[SliceState],
    rules: NetworkRules,
) -> dict:
    """Assemble the plain-data state passed into the prompt.
    See notebooks/01_agent_overview.ipynb, Section 2, for what is and isn't included.
    """
    return {
        "timestep": timestep,
        "rules": {
            "total_capacity_mbps": rules.total_capacity_mbps,
            "urllc_min_guarantee_mbps": rules.urllc_min_guarantee_mbps,
            "max_step_change_mbps": rules.max_step_change_mbps,
        },
        "slices": [
            {
                "name": s.name,
                "current_allocation_mbps": s.current_allocation_mbps,
                "observed_latency_ms": s.observed_latency_ms,
                "observed_throughput_mbps": s.observed_throughput_mbps,
                "forecast_mean_mbps": s.forecast_mean_mbps,
                "forecast_std_mbps": s.forecast_std_mbps,
            }
            for s in slice_states
        ],
    }


def build_prompt(state_context: dict, previous_error: Optional[str] = None) -> str:
    """Build the user-turn prompt. If `previous_error` is set, this is a
    retry -- the exact validation error is appended so the model can self-correct
    (notebooks/01_agent_overview.ipynb, Section 6).
    """
    prompt = (
        "Current network state (JSON):\n"
        f"{json.dumps(state_context, indent=2)}\n\n"
        "Return the allocation plan for the next timestep as specified."
    )
    if previous_error:
        prompt += (
            "\n\nYour previous response was REJECTED by the safety layer with this "
            f"exact error: \"{previous_error}\". Correct this and respond again with "
            "ONLY a valid JSON object."
        )
    return prompt


def _parse_llm_json(raw_text: str) -> Optional[dict]:
    """Best-effort parse of the LLM's raw text response into a dict.
    Returns None (not an exception) on failure -- the caller treats that as
    a Stage-A safety failure, exactly like a schema mismatch.
    """
    try:
        return json.loads(raw_text.strip())
    except (json.JSONDecodeError, AttributeError):
        return None


def plan_with_retry(
    timestep: int,
    slice_states: List[SliceState],
    rules: NetworkRules,
    previous_allocations: Dict[str, float],
    llm_call_fn: LLMCallFn,
    max_retries: int = MAX_RETRIES,
) -> AllocationPlan:
    """The bounded retry loop described in notebooks/01_agent_overview.ipynb, Section 6.

    Calls `llm_call_fn(system_prompt, user_prompt) -> raw_text` up to
    `1 + max_retries` times. On the first plan that passes both safety
    checks, returns it immediately. If every attempt fails, returns the
    deterministic fallback plan instead -- never raises on a bad LLM output.
    """
    state_context = build_state_context(timestep, slice_states, rules)
    previous_error: Optional[str] = None

    for attempt in range(1 + max_retries):
        user_prompt = build_prompt(state_context, previous_error)
        raw_text = llm_call_fn(SYSTEM_PROMPT, user_prompt)
        raw_dict = _parse_llm_json(raw_text)

        result = run_safety_checks(raw_dict, rules, previous_allocations)
        if result.ok:
            return AllocationPlan.from_dict(raw_dict)

        previous_error = result.error  # fed back into the next attempt's prompt

    # All attempts exhausted -- fall back to the deterministic safe policy.
    return fallback_policy(timestep, slice_states, rules)


# ---------------------------------------------------------------------------
# Self-test using a fake LLM (no API key / network access required).
# Run directly with: python3 src/agent_planner.py
# This exercises build_prompt -> parse -> safety checks -> retry loop end to
# end, which is exactly what Stage 4 will do with a real model swapped in.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rules = NetworkRules(
        total_capacity_mbps=100.0,
        urllc_min_guarantee_mbps=30.0,
        max_step_change_mbps=20.0,
    )
    slice_states = [
        SliceState("URLLC", current_allocation_mbps=32.0, observed_latency_ms=2.1,
                   observed_throughput_mbps=31.0, forecast_mean_mbps=38.0, forecast_std_mbps=5.0),
        SliceState("eMBB", current_allocation_mbps=68.0, observed_latency_ms=15.0,
                   observed_throughput_mbps=64.0, forecast_mean_mbps=55.0, forecast_std_mbps=9.0),
    ]
    previous_allocations = {"URLLC": 32.0, "eMBB": 68.0}

    # A fake "LLM" that first returns something over-capacity (fails Stage B),
    # then corrects itself on retry -- simulating exactly the self-correction
    # loop described in Section 6.
    call_count = {"n": 0}

    def fake_llm_call(system_prompt: str, user_prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Deliberately invalid: sums to 130 Mbps against a 100 Mbps capacity.
            return json.dumps({
                "timestep": 41,
                "allocations": {"URLLC": 40.0, "eMBB": 90.0},
                "reasoning": "Bumping both slices up to be safe.",
                "risk_flag": "low",
            })
        return json.dumps({
            "timestep": 41,
            "allocations": {"URLLC": 40.0, "eMBB": 60.0},
            "reasoning": "Corrected after safety-layer rejection: reduced eMBB to stay within capacity.",
            "risk_flag": "medium",
        })

    plan = plan_with_retry(
        timestep=41,
        slice_states=slice_states,
        rules=rules,
        previous_allocations=previous_allocations,
        llm_call_fn=fake_llm_call,
    )

    print(f"LLM calls made: {call_count['n']}")
    print("Final accepted plan:")
    print(json.dumps(plan.to_dict(), indent=2))
    assert call_count["n"] == 2, "expected exactly one retry"
    assert plan.allocations["URLLC"] + plan.allocations["eMBB"] <= rules.total_capacity_mbps
    print("\nSelf-test passed: bad plan was rejected, retry corrected it, final plan is within capacity.")
