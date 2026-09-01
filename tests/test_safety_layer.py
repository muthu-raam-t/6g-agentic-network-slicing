"""
Run with: pytest tests/ -v   (from the repo root, with src/ on PYTHONPATH)
e.g.:     PYTHONPATH=src pytest tests/ -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import AllocationPlan, NetworkRules, SliceState
from safety_layer import fallback_policy, validate_constraints, validate_schema
from agent_planner import plan_with_retry


RULES = NetworkRules(total_capacity_mbps=100.0, urllc_min_guarantee_mbps=30.0, max_step_change_mbps=20.0)
SLICE_STATES = [
    SliceState("URLLC", 32.0, 2.1, 31.0, 38.0, 5.0),
    SliceState("eMBB", 68.0, 15.0, 64.0, 55.0, 9.0),
]
PREV_ALLOC = {"URLLC": 32.0, "eMBB": 68.0}


def test_validate_schema_rejects_missing_field():
    result = validate_schema({"timestep": 1, "allocations": {"URLLC": 30}})
    assert not result.ok
    assert result.stage == "schema"


def test_validate_schema_accepts_well_formed_plan():
    result = validate_schema({
        "timestep": 1,
        "allocations": {"URLLC": 30.0, "eMBB": 70.0},
        "reasoning": "fine",
    })
    assert result.ok


def test_validate_constraints_rejects_over_capacity():
    plan = AllocationPlan(timestep=1, allocations={"URLLC": 40.0, "eMBB": 90.0}, reasoning="x")
    result = validate_constraints(plan, RULES, PREV_ALLOC)
    assert not result.ok
    assert "capacity" in result.error


def test_validate_constraints_rejects_urllc_below_minimum():
    plan = AllocationPlan(timestep=1, allocations={"URLLC": 10.0, "eMBB": 50.0}, reasoning="x")
    result = validate_constraints(plan, RULES, PREV_ALLOC)
    assert not result.ok
    assert "minimum guarantee" in result.error


def test_validate_constraints_rejects_oversized_step_change():
    plan = AllocationPlan(timestep=1, allocations={"URLLC": 60.0, "eMBB": 40.0}, reasoning="x")
    result = validate_constraints(plan, RULES, PREV_ALLOC)
    assert not result.ok
    assert "step" in result.error


def test_validate_constraints_accepts_valid_plan():
    plan = AllocationPlan(timestep=1, allocations={"URLLC": 40.0, "eMBB": 60.0}, reasoning="x")
    result = validate_constraints(plan, RULES, PREV_ALLOC)
    assert result.ok


def test_fallback_policy_always_within_capacity_and_meets_urllc_minimum():
    plan = fallback_policy(1, SLICE_STATES, RULES)
    total = sum(plan.allocations.values())
    assert total <= RULES.total_capacity_mbps
    assert plan.allocations["URLLC"] >= RULES.urllc_min_guarantee_mbps


def test_plan_with_retry_recovers_from_one_bad_attempt():
    calls = {"n": 0}

    def flaky_llm(system_prompt, user_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"timestep": 1, "allocations": {"URLLC": 40.0, "eMBB": 90.0}, "reasoning": "bad"})
        return json.dumps({"timestep": 1, "allocations": {"URLLC": 40.0, "eMBB": 60.0}, "reasoning": "fixed"})

    plan = plan_with_retry(1, SLICE_STATES, RULES, PREV_ALLOC, flaky_llm)
    assert calls["n"] == 2
    assert sum(plan.allocations.values()) <= RULES.total_capacity_mbps


def test_plan_with_retry_falls_back_when_llm_never_recovers():
    def always_broken_llm(system_prompt, user_prompt):
        return "not json at all"

    plan = plan_with_retry(1, SLICE_STATES, RULES, PREV_ALLOC, always_broken_llm)
    assert "Fallback policy" in plan.reasoning
    assert plan.allocations["URLLC"] == RULES.urllc_min_guarantee_mbps
