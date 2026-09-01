"""
baseline_agents.py
==================
STATUS: not yet built. Built out in notebooks/06_baseline_comparison.ipynb (Stage 6).

Will contain:
    - StaticBaselineAgent   -- fixed bandwidth split, never adapts (control arm)
    - LegacyPPOAgent        -- reloads/wraps the earlier semester's trained PPO
                               policy so it can be run as a second comparison arm
    - run_comparison(...)   -- runs all three arms (static / legacy RL / live
                               agentic) against the same twin + traffic and
                               collects P99 latency, QoS violation rate, and
                               eMBB throughput trade-off

See notebooks/00_overview.ipynb, Section 3 and notebooks/01_agent_overview.ipynb,
Section 9 for why this three-way comparison (not just agent-vs-baseline) is
the whole point of the final evaluation.
"""

raise NotImplementedError(
    "baseline_agents.py is a Stage 6 placeholder -- see notebooks/06_baseline_comparison.ipynb"
)
