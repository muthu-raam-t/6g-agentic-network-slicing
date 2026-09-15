import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digital_twin import DigitalTwin, SliceTrafficSpec
from schemas import NetworkRules
from baseline_agents import ReactiveLegacyAgent, StaticBaselineAgent, run_comparison


RULES = NetworkRules(total_capacity_mbps=100.0, urllc_min_guarantee_mbps=30.0, max_step_change_mbps=20.0)
SPECS = [
    SliceTrafficSpec("URLLC", base_demand_mbps=22.0, noise_std_mbps=1.5, spike_probability=0.03,
                      spike_multiplier=1.9, spike_decay=0.65),
    SliceTrafficSpec("eMBB", base_demand_mbps=45.0, noise_std_mbps=6.0, spike_probability=0.03,
                      spike_multiplier=2.2, spike_decay=0.7),
]


def make_twin():
    return DigitalTwin(SPECS, fading_rho=0.9, fading_min_fraction=0.8, contention_strength=0.12, seed=7)


def test_static_baseline_never_changes_allocation():
    agent = StaticBaselineAgent({"URLLC": 45.0, "eMBB": 55.0})
    a1 = agent.decide(0, [], RULES, {"URLLC": 50.0, "eMBB": 50.0})
    a2 = agent.decide(5, [{}], RULES, {"URLLC": 45.0, "eMBB": 55.0})
    assert a1 == {"URLLC": 45.0, "eMBB": 55.0}
    assert a2 == {"URLLC": 45.0, "eMBB": 55.0}


def test_reactive_legacy_agent_respects_urllc_minimum_and_capacity():
    agent = ReactiveLegacyAgent()
    twin = make_twin()
    allocations = {"URLLC": 50.0, "eMBB": 50.0}
    history = []
    for t in range(30):
        allocations = agent.decide(t, history, RULES, allocations)
        assert allocations["URLLC"] >= RULES.urllc_min_guarantee_mbps - 1e-6
        assert sum(allocations.values()) <= RULES.total_capacity_mbps + 1e-6
        obs = twin.step(allocations)
        history.append(obs)


def test_reactive_legacy_agent_shifts_toward_higher_latency_slice():
    """Core claim: if URLLC's latency history is much worse than eMBB's,
    the next decision should shift bandwidth toward URLLC relative to a
    neutral 50/50 starting point."""
    agent = ReactiveLegacyAgent()

    from digital_twin import SliceObservation
    skewed_history = [{
        "URLLC": SliceObservation(demand_mbps=30, latency_ms=40.0, throughput_mbps=25, effective_capacity_mbps=25),
        "eMBB": SliceObservation(demand_mbps=50, latency_ms=2.0, throughput_mbps=50, effective_capacity_mbps=55),
    }]
    allocations = agent.decide(1, skewed_history, RULES, {"URLLC": 50.0, "eMBB": 50.0})
    assert allocations["URLLC"] > 50.0


def test_run_comparison_returns_expected_metric_keys():
    controllers = {
        "static": StaticBaselineAgent({"URLLC": 45.0, "eMBB": 55.0}).decide,
        "reactive_legacy": ReactiveLegacyAgent().decide,
    }
    results = run_comparison(make_twin, controllers, RULES, num_steps=60)

    assert set(results.keys()) == {"static", "reactive_legacy"}
    for name, metrics in results.items():
        assert "p99_urllc_latency_ms" in metrics
        assert "qos_violation_rate" in metrics
        assert "mean_embb_throughput_mbps" in metrics
        assert len(metrics["urllc_latency_series"]) == 60
        assert 0.0 <= metrics["qos_violation_rate"] <= 1.0


def test_run_comparison_uses_independent_twins_per_controller():
    """Each controller must run against its own fresh twin instance (same
    seed/config), not a shared, already-stepped one."""
    call_count = {"n": 0}

    def counting_twin_factory():
        call_count["n"] += 1
        return make_twin()

    controllers = {"static": StaticBaselineAgent({"URLLC": 45.0, "eMBB": 55.0}).decide}
    run_comparison(counting_twin_factory, controllers, RULES, num_steps=5)
    assert call_count["n"] == 1

    controllers_two = {
        "static": StaticBaselineAgent({"URLLC": 45.0, "eMBB": 55.0}).decide,
        "reactive_legacy": ReactiveLegacyAgent().decide,
    }
    call_count["n"] = 0
    run_comparison(counting_twin_factory, controllers_two, RULES, num_steps=5)
    assert call_count["n"] == 2
