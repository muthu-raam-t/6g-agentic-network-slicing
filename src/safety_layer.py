"""
safety_layer.py
================
The deterministic guard between the LLM planner and actuation.

Design walkthrough: notebooks/01_agent_overview.ipynb, Sections 5 and 7.
This module has NO LLM calls and NO randomness on purpose -- safety cannot
depend on model behaviour. Every function here is a pure, testable function
of its inputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from schemas import AllocationPlan, NetworkRules, SliceState, ValidationResult

REQUIRED_FIELDS = {"timestep", "allocations", "reasoning"}


def validate_schema(raw: dict) -> ValidationResult:
    """Stage A -- syntactic check.

    Confirms `raw` (an already-JSON-decoded dict from the LLM response) has
    every required field, with the right types, before we even try to build
    an AllocationPlan out of it.
    """
    if not isinstance(raw, dict):
        return ValidationResult(False, "response is not a JSON object", "schema")

    missing = REQUIRED_FIELDS - raw.keys()
    if missing:
        return ValidationResult(False, f"missing required field(s): {sorted(missing)}", "schema")

    if not isinstance(raw["timestep"], int):
        return ValidationResult(False, "'timestep' must be an integer", "schema")

    if not isinstance(raw["allocations"], dict) or not raw["allocations"]:
        return ValidationResult(False, "'allocations' must be a non-empty object", "schema")

    for slice_name, value in raw["allocations"].items():
        if not isinstance(value, (int, float)):
            return ValidationResult(
                False, f"allocation for '{slice_name}' must be numeric, got {type(value).__name__}", "schema"
            )

    if not isinstance(raw["reasoning"], str) or not raw["reasoning"].strip():
        return ValidationResult(False, "'reasoning' must be a non-empty string", "schema")

    try:
        AllocationPlan.from_dict(raw)
    except (KeyError, ValueError, TypeError) as exc:
        return ValidationResult(False, f"could not build AllocationPlan: {exc}", "schema")

    return ValidationResult(True)


def validate_constraints(
    plan: AllocationPlan,
    rules: NetworkRules,
    previous_allocations: Dict[str, float],
    urllc_slice_name: str = "URLLC",
) -> ValidationResult:
    """Stage B -- semantic / physical constraint check.

    A plan that is syntactically perfect can still be physically nonsensical.
    Checks, in order (see notebooks/01_agent_overview.ipynb, Section 5B):
      1. no negative allocations
      2. total allocated <= total capacity
      3. URLLC minimum guarantee respected
      4. per-slice change from the previous timestep within the step-change bound
    """
    for slice_name, mbps in plan.allocations.items():
        if mbps < 0:
            return ValidationResult(
                False, f"allocation for '{slice_name}' is negative ({mbps} Mbps)", "constraints"
            )

    total = sum(plan.allocations.values())
    if total > rules.total_capacity_mbps + 1e-6:
        return ValidationResult(
            False,
            f"allocations sum to {total:.2f} Mbps, capacity is {rules.total_capacity_mbps:.2f} Mbps",
            "constraints",
        )

    urllc_alloc = plan.allocations.get(urllc_slice_name)
    if urllc_alloc is None:
        return ValidationResult(False, f"plan omits required slice '{urllc_slice_name}'", "constraints")
    if urllc_alloc < rules.urllc_min_guarantee_mbps - 1e-6:
        return ValidationResult(
            False,
            f"URLLC allocation {urllc_alloc:.2f} Mbps is below the minimum guarantee "
            f"of {rules.urllc_min_guarantee_mbps:.2f} Mbps",
            "constraints",
        )

    for slice_name, mbps in plan.allocations.items():
        prev = previous_allocations.get(slice_name)
        if prev is None:
            continue  # new slice, nothing to compare against
        step_change = abs(mbps - prev)
        if step_change > rules.max_step_change_mbps + 1e-6:
            return ValidationResult(
                False,
                f"'{slice_name}' would change by {step_change:.2f} Mbps in one step "
                f"(max allowed is {rules.max_step_change_mbps:.2f} Mbps)",
                "constraints",
            )

    return ValidationResult(True)


def fallback_policy(
    timestep: int,
    slice_states: List[SliceState],
    rules: NetworkRules,
    urllc_slice_name: str = "URLLC",
) -> AllocationPlan:
    """The deterministic, always-safe allocation used when every retry is
    exhausted (notebooks/01_agent_overview.ipynb, Section 7).

    Strategy: give URLLC exactly its minimum guarantee, split the remaining
    capacity proportionally across the other slices by their forecast mean
    demand. This always satisfies validate_constraints() by construction,
    given consistent `rules`.
    """
    remaining = rules.total_capacity_mbps - rules.urllc_min_guarantee_mbps
    other_slices = [s for s in slice_states if s.name != urllc_slice_name]
    total_other_demand = sum(max(s.forecast_mean_mbps, 0.0) for s in other_slices) or 1.0

    allocations = {urllc_slice_name: rules.urllc_min_guarantee_mbps}
    for s in other_slices:
        share = max(s.forecast_mean_mbps, 0.0) / total_other_demand
        allocations[s.name] = round(remaining * share, 2)

    return AllocationPlan(
        timestep=timestep,
        allocations=allocations,
        reasoning=(
            "Fallback policy: LLM plan failed validation after all retries. "
            "URLLC held at its minimum guarantee; remaining capacity split "
            "proportionally to forecast demand across other slices."
        ),
    )


def run_safety_checks(
    raw_llm_output: Optional[dict],
    rules: NetworkRules,
    previous_allocations: Dict[str, float],
    urllc_slice_name: str = "URLLC",
) -> ValidationResult:
    """Convenience wrapper: Stage A then Stage B, short-circuiting on the first
    failure. Used directly by `agent_planner.plan_with_retry()`.
    """
    if raw_llm_output is None:
        return ValidationResult(False, "LLM returned no parseable output", "schema")

    schema_result = validate_schema(raw_llm_output)
    if not schema_result.ok:
        return schema_result

    plan = AllocationPlan.from_dict(raw_llm_output)
    return validate_constraints(plan, rules, previous_allocations, urllc_slice_name)
