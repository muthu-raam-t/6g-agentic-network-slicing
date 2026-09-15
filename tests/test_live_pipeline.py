import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digital_twin import DigitalTwin, SliceTrafficSpec
from schemas import NetworkRules
from baseline_agents import StaticBaselineAgent, ReactiveLegacyAgent, run_comparison
from reference_llm import build_reference_llm
from live_pipeline import make_live_agentic_controller

RULES = NetworkRules(total_capacity_mbps=100.0, urllc_min_guarantee_mbps=30.0, max_step_change_mbps=20.0)
SPECS = [
    SliceTrafficSpec("URLLC", base_demand_mbps=22.0, noise_std_mbps=1.5, spike_probability=0.03,
                      spike_multiplier=1.9, spike_decay=0.65),
    SliceTrafficSpec("eMBB", base_demand_mbps=45.0, noise_std_mbps=6.0, spike_probability=0.03,
                      spike_multiplier=2.2, spike_decay=0.7),
]


def make_twin():
    return DigitalTwin(SPECS, fading_rho=0.9, fading_min_fraction=0.8, contention_strength=0.12, seed=7)


def test_live_agentic_controller_returns_previous_allocation_during_warmup():
    controller = make_live_agentic_controller(build_reference_llm())
    prev = {"URLLC": 45.0, "eMBB": 55.0}
    result = controller(0, [], RULES, prev)
    assert result == prev


def test_live_agentic_controller_respects_constraints_over_a_run():
    controller = make_live_agentic_controller(build_reference_llm())
    twin = make_twin()
    allocations = {"URLLC": 50.0, "eMBB": 50.0}
    history = []
    for t in range(60):
        allocations = controller(t, history, RULES, allocations)
        assert allocations["URLLC"] >= RULES.urllc_min_guarantee_mbps - 1e-6
        assert sum(allocations.values()) <= RULES.total_capacity_mbps + 1e-6
        obs = twin.step(allocations)
        history.append(obs)


def test_full_three_arm_comparison_agentic_beats_both_baselines_on_qos():
    """The headline empirical claim of the project: the live agentic system
    (tuned reference LLM) should achieve a LOWER QoS violation rate than
    both the static baseline and the reactive legacy agent, on identical
    traffic. This is what notebooks/06_baseline_comparison.ipynb reports.
    """
    controllers = {
        "static": StaticBaselineAgent({"URLLC": 45.0, "eMBB": 60.0}).decide,
        "reactive_legacy": ReactiveLegacyAgent().decide,
        "live_agentic": make_live_agentic_controller(build_reference_llm()),
    }
    results = run_comparison(make_twin, controllers, RULES, num_steps=300)

    static_violation = results["static"]["qos_violation_rate"]
    reactive_violation = results["reactive_legacy"]["qos_violation_rate"]
    agentic_violation = results["live_agentic"]["qos_violation_rate"]

    assert agentic_violation < static_violation
    assert agentic_violation < reactive_violation

    # The eMBB throughput trade-off should be real but not catastrophic --
    # the agentic system shouldn't be starving eMBB down near zero.
    assert results["live_agentic"]["mean_embb_throughput_mbps"] > 20.0
