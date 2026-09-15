"""
digital_twin.py
================
Stage 2: the simulated network the forecaster predicts and the planner controls.

Design walkthrough: notebooks/02_digital_twin.ipynb.
Deliberately NOT trivial to predict or control -- see the three mechanisms
below -- so that the forecaster/planner stages later in the pipeline are
solving a real problem, not a toy one.

Three behaviours modelled, matching notebooks/00_overview.ipynb, Section 1:
    1. Non-linear "congestion cliff" latency (congestion_cliff_delay)
    2. Stateful, correlated wireless fading (CorrelatedFadingChannel)
    3. Shared base-station resource contention (applied inside DigitalTwin.step)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Congestion-cliff latency model
# ---------------------------------------------------------------------------

def congestion_cliff_delay(
    load_mbps: float,
    capacity_mbps: float,
    floor_ms: float = 1.0,
    scale: float = 15.0,
    exponent: float = 11 / 5,
    overload_penalty: float = 120.0,
) -> float:
    """Queuing delay as a function of load vs. allocated capacity.

    Stays near `floor_ms` while load is well under capacity, then rises
    sharply as load approaches capacity (the "cliff"), matching
    notebooks/00_overview.ipynb, Section 4.1. If load actually exceeds
    capacity (a real overload, e.g. from a bad allocation), an additional
    linear overload penalty is added on top so the model doesn't just
    plateau -- overload should look and behave clearly worse than "near
    the cliff".
    """
    if capacity_mbps <= 0:
        return float("inf")

    ratio = load_mbps / capacity_mbps
    clipped_ratio = min(ratio, 1.0)
    delay = floor_ms + scale * clipped_ratio ** exponent

    if ratio > 1.0:
        delay += overload_penalty * (ratio - 1.0)

    return delay


# ---------------------------------------------------------------------------
# 2. Correlated wireless fading channel (stateful, not white noise)
# ---------------------------------------------------------------------------

class CorrelatedFadingChannel:
    """An AR(1) process representing slowly-varying wireless channel quality.

    Real fading is correlated in time -- a bad channel tends to stay bad for
    a while, not flicker randomly every timestep. Modelled as:

        state_t = rho * state_{t-1} + sqrt(1 - rho^2) * noise_t,   noise_t ~ N(0, 1)

    which keeps state_t roughly standard-normal-distributed at all times
    (its stationary variance is 1) while `rho` controls how correlated
    consecutive steps are.

    `effective_capacity_fraction()` maps the (unbounded) state to a fraction
    in (min_fraction, 1.0] via a logistic squashing function, representing
    how much of the nominal slice capacity is actually usable this timestep.
    """

    def __init__(self, rho: float = 0.9, min_fraction: float = 0.6, seed: Optional[int] = None):
        if not (0.0 <= rho < 1.0):
            raise ValueError("rho must be in [0, 1)")
        self.rho = rho
        self.min_fraction = min_fraction
        self._rng = random.Random(seed)
        self.state = 0.0

    def step(self) -> float:
        """Advance the channel by one timestep and return the new effective-
        capacity fraction."""
        noise = self._rng.gauss(0.0, 1.0)
        self.state = self.rho * self.state + math.sqrt(1 - self.rho ** 2) * noise
        return self.effective_capacity_fraction()

    def effective_capacity_fraction(self) -> float:
        """Squash the current state into (min_fraction, 1.0] with a logistic
        curve, so an average state (~0) gives a fraction near the midpoint
        and a deeply faded state approaches min_fraction, never zero."""
        logistic = 1.0 / (1.0 + math.exp(-self.state))  # in (0, 1)
        return self.min_fraction + (1.0 - self.min_fraction) * logistic


# ---------------------------------------------------------------------------
# 3. Traffic demand generator (per slice, with occasional spikes)
# ---------------------------------------------------------------------------

@dataclass
class SliceTrafficSpec:
    name: str
    base_demand_mbps: float
    noise_std_mbps: float
    spike_probability: float = 0.02
    spike_multiplier: float = 2.2
    spike_decay: float = 0.75  # how quickly a spike fades back to baseline, per step


class TrafficGenerator:
    """Generates a per-slice demand series with a baseline, Gaussian noise,
    and occasional multi-step spikes that decay geometrically -- so a spike
    is a short, forecastable-in-principle *event*, not a single-step outlier,
    which is what gives the forecaster's uncertainty estimate (Section 5,
    00_overview.ipynb) something meaningful to react to.
    """

    def __init__(self, specs: List[SliceTrafficSpec], seed: Optional[int] = None):
        self.specs = {s.name: s for s in specs}
        self._rng = random.Random(seed)
        self._active_spike: Dict[str, float] = {s.name: 0.0 for s in specs}

    def step(self) -> Dict[str, float]:
        demand = {}
        for name, spec in self.specs.items():
            if self._active_spike[name] < 1e-3 and self._rng.random() < spec.spike_probability:
                self._active_spike[name] = spec.spike_multiplier - 1.0  # extra fraction above baseline

            extra = self._active_spike[name]
            self._active_spike[name] *= spec.spike_decay  # decays every step, including the step it starts

            noise = self._rng.gauss(0.0, spec.noise_std_mbps)
            demand[name] = max(0.0, spec.base_demand_mbps * (1.0 + extra) + noise)

        return demand


# ---------------------------------------------------------------------------
# Digital Twin: ties all three mechanisms together into one step function
# ---------------------------------------------------------------------------

@dataclass
class SliceObservation:
    demand_mbps: float
    latency_ms: float
    throughput_mbps: float
    effective_capacity_mbps: float


class DigitalTwin:
    """The full simulated network for one control loop.

    `step(allocations)` takes the planner's chosen bandwidth split and
    returns, per slice: the traffic that actually showed up, the resulting
    latency (via the congestion cliff, against a capacity reduced by both
    fading AND contention), and observed throughput.

    Shared contention (mechanism 3): each slice's *effective* capacity is
    reduced not only by its own fading but by a contention penalty driven by
    how heavily loaded the *other* slices are -- a busy eMBB slice quietly
    eats into URLLC's real capacity even though URLLC's nominal allocation
    didn't change, which is exactly the failure mode a naive controller
    (one that only watches its own slice) would miss.
    """

    def __init__(
        self,
        traffic_specs: List[SliceTrafficSpec],
        fading_rho: float = 0.9,
        fading_min_fraction: float = 0.6,
        contention_strength: float = 0.25,
        seed: Optional[int] = None,
    ):
        self.slice_names = [s.name for s in traffic_specs]
        self.traffic_gen = TrafficGenerator(traffic_specs, seed=seed)
        self.fading_channels = {
            name: CorrelatedFadingChannel(rho=fading_rho, min_fraction=fading_min_fraction,
                                           seed=None if seed is None else seed + i)
            for i, name in enumerate(self.slice_names)
        }
        self.contention_strength = contention_strength
        self.timestep = 0

    def _contention_penalty(self, allocations: Dict[str, float], focus_slice: str) -> float:
        """Fraction (0..~contention_strength) by which focus_slice's effective
        capacity is reduced due to load on every OTHER slice."""
        other_total = sum(v for name, v in allocations.items() if name != focus_slice)
        total = sum(allocations.values()) or 1.0
        other_share = other_total / total
        return self.contention_strength * other_share

    def step(self, allocations: Dict[str, float]) -> Dict[str, SliceObservation]:
        """Advance the twin by one timestep.

        `allocations`: the bandwidth (Mbps) given to each slice this
        timestep, as produced by whichever controller (static / legacy RL /
        live agentic planner) is being evaluated.
        """
        self.timestep += 1
        demands = self.traffic_gen.step()

        observations: Dict[str, SliceObservation] = {}
        for name in self.slice_names:
            nominal_capacity = allocations.get(name, 0.0)
            fading_fraction = self.fading_channels[name].step()
            contention_frac = self._contention_penalty(allocations, name)

            effective_capacity = nominal_capacity * fading_fraction * (1.0 - contention_frac)
            effective_capacity = max(effective_capacity, 1e-6)

            demand = demands[name]
            latency = congestion_cliff_delay(demand, effective_capacity)
            throughput = min(demand, effective_capacity)

            observations[name] = SliceObservation(
                demand_mbps=demand,
                latency_ms=latency,
                throughput_mbps=throughput,
                effective_capacity_mbps=effective_capacity,
            )

        return observations

    def reset(self):
        """Reset timestep counter only -- construct a new DigitalTwin for a
        fully independent run (fresh RNG state)."""
        self.timestep = 0
