import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digital_twin import (
    CorrelatedFadingChannel,
    DigitalTwin,
    SliceTrafficSpec,
    TrafficGenerator,
    congestion_cliff_delay,
)


def test_congestion_cliff_low_load_is_near_floor():
    delay = congestion_cliff_delay(load_mbps=10.0, capacity_mbps=100.0, floor_ms=1.0)
    assert 1.0 <= delay < 2.0  # far from the cliff, should sit near the floor


def test_congestion_cliff_is_monotonically_increasing():
    loads = [10, 30, 50, 70, 90, 99]
    delays = [congestion_cliff_delay(l, 100.0) for l in loads]
    assert all(delays[i] < delays[i + 1] for i in range(len(delays) - 1))


def test_congestion_cliff_overload_is_penalized_beyond_the_plateau():
    at_capacity = congestion_cliff_delay(100.0, 100.0)
    over_capacity = congestion_cliff_delay(150.0, 100.0)
    assert over_capacity > at_capacity + 20  # overload penalty should bite, clearly


def test_congestion_cliff_zero_capacity_is_infinite():
    assert congestion_cliff_delay(10.0, 0.0) == float("inf")


def test_fading_channel_stays_within_bounds():
    channel = CorrelatedFadingChannel(rho=0.9, min_fraction=0.6, seed=1)
    fractions = [channel.step() for _ in range(500)]
    assert all(0.6 <= f <= 1.0 for f in fractions)


def test_fading_channel_is_correlated_not_white_noise():
    """A high-rho AR(1) process should have much smaller step-to-step jumps
    than pure independent noise would -- checks the *correlation* behaviour,
    not just that values are in range."""
    correlated = CorrelatedFadingChannel(rho=0.95, seed=2)
    correlated_series = [correlated.step() for _ in range(300)]
    correlated_jumps = [abs(correlated_series[i + 1] - correlated_series[i]) for i in range(len(correlated_series) - 1)]

    independent = CorrelatedFadingChannel(rho=0.0, seed=2)
    independent_series = [independent.step() for _ in range(300)]
    independent_jumps = [abs(independent_series[i + 1] - independent_series[i]) for i in range(len(independent_series) - 1)]

    avg_correlated_jump = sum(correlated_jumps) / len(correlated_jumps)
    avg_independent_jump = sum(independent_jumps) / len(independent_jumps)
    assert avg_correlated_jump < avg_independent_jump


def test_traffic_generator_produces_nonnegative_demand():
    specs = [SliceTrafficSpec("URLLC", base_demand_mbps=30.0, noise_std_mbps=5.0)]
    gen = TrafficGenerator(specs, seed=3)
    for _ in range(200):
        demand = gen.step()
        assert demand["URLLC"] >= 0.0


def test_twin_step_returns_all_slices():
    specs = [
        SliceTrafficSpec("URLLC", base_demand_mbps=30.0, noise_std_mbps=3.0),
        SliceTrafficSpec("eMBB", base_demand_mbps=55.0, noise_std_mbps=8.0),
    ]
    twin = DigitalTwin(specs, seed=4)
    obs = twin.step({"URLLC": 35.0, "eMBB": 65.0})
    assert set(obs.keys()) == {"URLLC", "eMBB"}
    for slice_obs in obs.values():
        assert slice_obs.latency_ms > 0
        assert slice_obs.throughput_mbps >= 0
        assert slice_obs.effective_capacity_mbps > 0


def test_contention_degrades_other_slices_when_one_slice_is_heavily_loaded():
    """Core claim of the shared-contention mechanism: giving eMBB a much
    larger share of total allocation should measurably reduce URLLC's
    effective capacity, even though URLLC's own nominal allocation is
    identical in both cases."""
    specs = [
        SliceTrafficSpec("URLLC", base_demand_mbps=20.0, noise_std_mbps=0.0, spike_probability=0.0),
        SliceTrafficSpec("eMBB", base_demand_mbps=20.0, noise_std_mbps=0.0, spike_probability=0.0),
    ]

    twin_low_contention = DigitalTwin(specs, fading_rho=0.0, contention_strength=0.25, seed=5)
    obs_low = twin_low_contention.step({"URLLC": 30.0, "eMBB": 5.0})

    twin_high_contention = DigitalTwin(specs, fading_rho=0.0, contention_strength=0.25, seed=5)
    obs_high = twin_high_contention.step({"URLLC": 30.0, "eMBB": 200.0})

    assert obs_high["URLLC"].effective_capacity_mbps < obs_low["URLLC"].effective_capacity_mbps


def test_twin_is_reproducible_with_same_seed():
    specs = [SliceTrafficSpec("URLLC", base_demand_mbps=30.0, noise_std_mbps=5.0)]
    twin_a = DigitalTwin(specs, seed=42)
    twin_b = DigitalTwin(specs, seed=42)
    obs_a = twin_a.step({"URLLC": 35.0})
    obs_b = twin_b.step({"URLLC": 35.0})
    assert obs_a["URLLC"].demand_mbps == obs_b["URLLC"].demand_mbps
    assert obs_a["URLLC"].latency_ms == obs_b["URLLC"].latency_ms
