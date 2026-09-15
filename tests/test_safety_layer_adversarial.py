"""
Adversarial tests for the safety layer: deliberately malformed, hostile, or
nonsensical LLM outputs, checked against two invariants that must hold no
matter what garbage comes in:
    1. run_safety_checks() / validate_schema() / validate_constraints()
       never raise -- they always return a ValidationResult.
    2. plan_with_retry() never raises and never returns a plan that violates
       the hard constraints, even when every retry attempt is also garbage.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import NetworkRules, SliceState
from safety_layer import run_safety_checks, validate_schema
from agent_planner import plan_with_retry

RULES = NetworkRules(total_capacity_mbps=100.0, urllc_min_guarantee_mbps=30.0, max_step_change_mbps=20.0)
PREV_ALLOC = {"URLLC": 40.0, "eMBB": 60.0}
SLICE_STATES = [
    SliceState("URLLC", 40.0, 5.0, 38.0, 42.0, 4.0),
    SliceState("eMBB", 60.0, 12.0, 58.0, 55.0, 8.0),
]

SCHEMA_LEVEL_ADVERSARIAL_OUTPUTS = [
    {},                                                     # empty object
    {"timestep": 1},                                        # missing everything else
    {"timestep": "one", "allocations": {"URLLC": 40}, "reasoning": "x"},  # wrong type for timestep
    {"timestep": 1, "allocations": "give it all to URLLC", "reasoning": "x"},  # allocations not a dict
    {"timestep": 1, "allocations": {}, "reasoning": "x"},    # empty allocations
    {"timestep": 1, "allocations": {"URLLC": "lots", "eMBB": 10}, "reasoning": "x"},  # non-numeric value
    {"timestep": 1, "allocations": {"URLLC": 40.0, "eMBB": 60.0}},      # missing reasoning field
    {"timestep": 1, "allocations": {"URLLC": 40.0, "eMBB": 60.0}, "reasoning": ""},  # empty reasoning string
    {"timestep": 1, "allocations": {"URLLC": float("nan"), "eMBB": 60.0}, "reasoning": "x"},  # NaN
    {"timestep": 1, "allocations": {"URLLC": float("inf"), "eMBB": 60.0}, "reasoning": "x"},  # inf
    {"timestep": 1, "allocations": {"URLLC": True, "eMBB": 60.0}, "reasoning": "x"},  # bool masquerading as numeric
    [1, 2, 3],                                               # a list, not an object at all
    "just a plain string, not even JSON-shaped",
    42,
]

ADVERSARIAL_RAW_OUTPUTS = SCHEMA_LEVEL_ADVERSARIAL_OUTPUTS + [
    None,                                                    # nothing parseable at all
    {"timestep": 1, "allocations": {"URLLC": -50.0, "eMBB": 60.0}, "reasoning": "x"},  # negative
    {"timestep": 1, "allocations": {"URLLC": 200.0, "eMBB": 200.0}, "reasoning": "x"},  # wildly over capacity
    {"timestep": 1, "allocations": {"URLLC": 5.0, "eMBB": 95.0}, "reasoning": "x"},  # below URLLC minimum
    {"timestep": 1, "allocations": {"eMBB": 60.0}, "reasoning": "x"},   # URLLC slice omitted entirely
    {"timestep": 1, "allocations": {"URLLC": 90.0, "eMBB": 10.0}, "reasoning": "x"},  # step change way too big
]


def test_run_safety_checks_never_raises_on_any_adversarial_input():
    for bad_output in ADVERSARIAL_RAW_OUTPUTS:
        result = run_safety_checks(bad_output, RULES, PREV_ALLOC)
        assert result.ok is False, f"expected rejection for: {bad_output!r}"
        assert isinstance(result.error, str) and result.error  # always carries a real message


def test_validate_schema_never_raises_on_any_schema_level_adversarial_input():
    for bad_output in SCHEMA_LEVEL_ADVERSARIAL_OUTPUTS:
        result = validate_schema(bad_output)
        assert result.ok is False, f"expected schema rejection for: {bad_output!r}"


def test_plan_with_retry_always_falls_back_when_every_attempt_is_garbage():
    """Simulates the worst case: the LLM returns different garbage on every
    single retry attempt. The system must never crash and must never act
    on any of it -- only the deterministic fallback is acceptable.
    """
    rng = random.Random(123)

    def chaotic_llm(system_prompt, user_prompt):
        choice = rng.choice(ADVERSARIAL_RAW_OUTPUTS)
        if isinstance(choice, (dict, list)):
            return json.dumps(choice)
        return str(choice)

    plan = plan_with_retry(1, SLICE_STATES, RULES, PREV_ALLOC, chaotic_llm)

    assert "Fallback policy" in plan.reasoning
    assert plan.allocations["URLLC"] >= RULES.urllc_min_guarantee_mbps
    assert sum(plan.allocations.values()) <= RULES.total_capacity_mbps + 1e-6


def test_plan_with_retry_survives_an_llm_that_raises_python_exceptions():
    """Even a callable that throws (e.g. a real API client hitting a network
    error) must not crash the whole control loop -- plan_with_retry() should
    let the exception surface immediately rather than silently swallow it,
    so the caller can decide whether to treat a network failure differently
    from a malformed-output failure. This test documents that current
    behaviour explicitly rather than leaving it implicit.
    """
    def broken_llm(system_prompt, user_prompt):
        raise ConnectionError("simulated network failure")

    try:
        plan_with_retry(1, SLICE_STATES, RULES, PREV_ALLOC, broken_llm)
        assert False, "expected the underlying ConnectionError to propagate"
    except ConnectionError:
        pass  # documented, current behaviour: caller is responsible for catching real API/network errors


def test_plan_with_retry_recovers_immediately_if_first_attempt_is_good():
    """Sanity check in the adversarial file too: a well-formed first attempt
    should need zero retries, even sitting next to all this chaos-testing.
    """
    calls = {"n": 0}

    def good_llm(system_prompt, user_prompt):
        calls["n"] += 1
        return json.dumps({
            "timestep": 1,
            "allocations": {"URLLC": 45.0, "eMBB": 55.0},
            "reasoning": "steady state",
        })

    plan = plan_with_retry(1, SLICE_STATES, RULES, PREV_ALLOC, good_llm)
    assert calls["n"] == 1
    assert "Fallback" not in plan.reasoning
