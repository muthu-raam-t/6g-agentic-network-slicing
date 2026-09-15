import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reference_llm import build_reference_llm
from schemas import NetworkRules, SliceState
from safety_layer import validate_constraints, validate_schema
from schemas import AllocationPlan

RULES = NetworkRules(total_capacity_mbps=100.0, urllc_min_guarantee_mbps=30.0, max_step_change_mbps=20.0)


def _call_reference_llm(llm, timestep, slice_states, previous_allocations):
    from agent_planner import build_state_context, build_prompt, SYSTEM_PROMPT
    state_context = build_state_context(timestep, slice_states, RULES)
    prompt = build_prompt(state_context)
    raw_text = llm(SYSTEM_PROMPT, prompt)
    return json.loads(raw_text)


def test_reference_llm_output_passes_schema_validation():
    llm = build_reference_llm()
    slice_states = [
        SliceState("URLLC", 45.0, 5.0, 22.0, 22.0, 2.0),
        SliceState("eMBB", 55.0, 12.0, 50.0, 48.0, 6.0),
    ]
    raw = _call_reference_llm(llm, 1, slice_states, {"URLLC": 45.0, "eMBB": 55.0})
    result = validate_schema(raw)
    assert result.ok, result.error


def test_reference_llm_output_passes_constraint_validation():
    llm = build_reference_llm()
    slice_states = [
        SliceState("URLLC", 45.0, 5.0, 22.0, 22.0, 2.0),
        SliceState("eMBB", 55.0, 12.0, 50.0, 48.0, 6.0),
    ]
    raw = _call_reference_llm(llm, 1, slice_states, {"URLLC": 45.0, "eMBB": 55.0})
    plan = AllocationPlan.from_dict(raw)
    result = validate_constraints(plan, RULES, {"URLLC": 45.0, "eMBB": 55.0})
    assert result.ok, result.error


def test_reference_llm_respects_urllc_minimum_even_with_low_forecast():
    llm = build_reference_llm()
    slice_states = [
        SliceState("URLLC", 30.0, 1.0, 5.0, 2.0, 0.1),  # tiny forecast, well below urllc_min
        SliceState("eMBB", 70.0, 5.0, 50.0, 48.0, 4.0),
    ]
    raw = _call_reference_llm(llm, 1, slice_states, {"URLLC": 30.0, "eMBB": 70.0})
    assert raw["allocations"]["URLLC"] >= RULES.urllc_min_guarantee_mbps


def test_reference_llm_increases_urllc_allocation_when_forecast_rises():
    llm = build_reference_llm()
    low_forecast = [
        SliceState("URLLC", 45.0, 5.0, 22.0, 15.0, 1.0),
        SliceState("eMBB", 55.0, 12.0, 50.0, 48.0, 6.0),
    ]
    high_forecast = [
        SliceState("URLLC", 45.0, 5.0, 22.0, 35.0, 1.0),
        SliceState("eMBB", 55.0, 12.0, 50.0, 48.0, 6.0),
    ]
    raw_low = _call_reference_llm(llm, 1, low_forecast, {"URLLC": 45.0, "eMBB": 55.0})
    raw_high = _call_reference_llm(llm, 1, high_forecast, {"URLLC": 45.0, "eMBB": 55.0})
    assert raw_high["allocations"]["URLLC"] > raw_low["allocations"]["URLLC"]
