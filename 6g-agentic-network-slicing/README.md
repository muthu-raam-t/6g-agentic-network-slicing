# Agentic AI for Proactive QoS Assurance in 6G Network Slicing

An autonomous, closed-loop control system that manages network resources in a simulated 6G environment — predicting traffic before it spikes, reasoning about the best allocation with an LLM, and verifying every decision through a safety layer before it ever touches the network.

## The Problem

6G networks use **network slicing** to split one physical network into multiple virtual ones, each serving a different type of traffic:

- **URLLC** (Ultra-Reliable Low-Latency Communication) — self-driving vehicles, remote surgery, industrial robotics. Even a few milliseconds of delay can cause real, physical failure.
- **eMBB** (Enhanced Mobile Broadband) — streaming, downloads. Can tolerate delay without serious consequences.

Traditional systems assign each slice a **fixed** amount of bandwidth. That works fine at low traffic, but the moment demand spikes unpredictably, the rigid split can't adapt — and the latency-critical slice (URLLC) breaches its QoS guarantee. This project builds a system that watches, predicts, and reallocates resources *before* that happens.

## Why 6G Specifically

- 6G's target latencies for URLLC push toward sub-millisecond, far tighter than 5G — the margin for error is smaller.
- "AI-native" self-managing networks are a core design goal of 6G, not an afterthought.
- Network Digital Twins are an actively researched 6G enabler for safely developing and testing this kind of AI before real deployment.

## Architecture

A continuous four-stage control loop, repeating every timestep:

```
Enhanced Digital Twin  →  Probabilistic Forecaster  →  Live Agentic Planner  →  Safety Layer  →  Actuation
        ↑_____________________________________________________________________________________|
```

**1. Enhanced Digital Twin**
A high-fidelity mathematical simulation of the network, built to be deliberately hard to fool an agent with. Models three realistic failure behaviors:
- Non-linear "congestion cliff" latency (stable until a threshold, then degrades sharply)
- Stateful, correlated wireless channel fading (not random noise)
- Shared base-station resource contention — heavy load on one slice quietly degrades all slices

**2. Probabilistic Forecaster (LSTM)**
Looks at recent traffic history per slice and predicts not just the next value, but how *uncertain* that prediction is (mean + variance). This uncertainty is what lets the planner make risk-aware decisions instead of reacting blindly.

**3. Live Agentic Planner (LLM)**
Given the current state, the forecast, and the network's hard rules, an LLM reasons through the situation like a human network operator and outputs a structured allocation plan — in plain, inspectable language, not opaque neural weights.

**4. Safety Layer**
Every plan is checked twice before it's allowed to act: (1) is it valid, well-formed output, and (2) does it violate any physical or operational constraint. If either check fails, the system falls back to a deterministic, known-safe policy for that timestep instead of acting on an unverified plan.

## Why Not Just Use Reinforcement Learning?

An earlier prototype of this project used a PPO reinforcement learning agent. It worked, but had two problems this architecture is built to avoid:
- **Opaque decisions** — an RL agent's choice can't be inspected or explained, only observed.
- **Purely reactive** — it only responds after congestion is already happening, not before.

This version replaces the RL agent entirely with the LLM-planner + forecaster + safety-layer pipeline above, trading a black box for a system that is proactive, explainable, and independently verified before it acts.

## Who This Is For

Not an end-user product — the "user" is the network itself, and by extension:
- **Telecom operators**, who would deploy this inside their network core to manage slicing decisions live.
- **Network Operations Center (NOC) engineers**, whose current manual/reactive monitoring role this is designed to assist or replace.
- **Indirectly**, the businesses depending on the URLLC guarantee — hospitals, factories, autonomous fleets — who never touch the system but are the reason it needs to work.

## What Success Looks Like

Measured by comparing this system against a static, fixed-allocation baseline under identical, unpredictable traffic:
- Reduction in **P99 tail latency** for the URLLC slice (worst-case user experience)
- Reduction in **QoS violation rate** (how often latency crosses the critical threshold)
- An honest, quantified trade-off in **eMBB throughput** — the acceptable cost of protecting URLLC

## Project Status

Currently being rebuilt from an earlier RL-based prototype into this full architecture. Build order:

1. Enhanced Digital Twin
2. Probabilistic LSTM Forecaster
3. Live LLM Agentic Planner
4. Safety Layer
5. Baseline vs. Agentic system comparison harness
6. Results visualization and reporting

## Roadmap / Possible Extensions

- Live dashboard showing real-time traffic, allocations, and QoS status
- An "agent reasoning" panel exposing the LLM's plain-English explanation for each decision
- A manual fault/spike injection control to demo agent vs. baseline behavior side by side

## Getting Started

```bash
git clone https://github.com/muthu-raam-t/6g-agentic-network-slicing.git
cd 6g-agentic-network-slicing
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/00_overview.ipynb
```

If you plan to run the live agentic planner (Stage 4 onward) against a real LLM, set:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Repository Structure

```
6g-agentic-network-slicing/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 00_overview.ipynb          <- start here
│   ├── 01_agent_overview.ipynb    <- agent design deep dive (explanation only)
│   ├── 02_digital_twin.ipynb
│   ├── 03_forecaster.ipynb
│   ├── 04_agentic_planner.ipynb
│   ├── 05_safety_layer.ipynb
│   ├── 06_baseline_comparison.ipynb
│   └── 07_results_and_report.ipynb
├── src/
│   ├── schemas.py
│   ├── safety_layer.py
│   ├── agent_planner.py
│   ├── digital_twin.py
│   ├── forecaster.py
│   └── baseline_agents.py
├── images/
│   └── system_architecture.svg
├── results/
└── data/
```

Each notebook after `00_overview.ipynb` holds explanation and math; the matching file in `src/` holds the real, runnable implementation.
