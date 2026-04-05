"""
reward/core.py

Computes per-step reward from state_before/state_after dicts and action.
Called by environment.py's _compute_reward() with this exact signature:

    compute_step_reward(
        state_before = {...},
        state_after  = {...},
        action       = Action(...),
        user_event   = "neutral" | "positive" | "frustrated" | ...,
        done         = True | False,
        reward_weights = {...},   # from TaskConfig, may be empty
    )
"""

from __future__ import annotations
from typing import Dict, Any, Tuple


def compute_step_reward(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    action,                          # models.Action — avoid circular import
    user_event: str,
    done: bool,
    reward_weights: Dict[str, float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total reward and per-component breakdown for one environment step.

    Parameters
    ----------
    state_before    : snapshot of hidden State fields before the action
    state_after     : snapshot of hidden State fields after the action
    action          : the Action taken (Pydantic object with .action_type)
    user_event      : string event from the user simulator
    done            : whether the episode terminated this step
    reward_weights  : optional per-task weight overrides (currently reserved)

    Returns
    -------
    (total_reward: float, components: Dict[str, float])
    """
    weights = reward_weights or {}

    components: Dict[str, float] = {
        "satisfaction_gain": 0.0,
        "annoyance_penalty": 0.0,
        "obligation_penalty": 0.0,
        "cost_penalty": 0.0,
        "stage_progress": 0.0,
        "delay_penalty": 0.0,
        "terminal": 0.0,
    }

    # ── satisfaction gain ─────────────────────────────────────────────────────
    sat_before = state_before.get("satisfaction", 0.5)
    sat_after  = state_after.get("satisfaction", 0.5)
    components["satisfaction_gain"] = sat_after - sat_before

    # ── annoyance penalty ─────────────────────────────────────────────────────
    ann_before = state_before.get("annoyance", 0.0)
    ann_after  = state_after.get("annoyance", 0.0)
    annoyance_delta = ann_after - ann_before
    components["annoyance_penalty"] = -annoyance_delta   # negative when annoyance rises

    # ── obligation violation penalty ──────────────────────────────────────────
    violations = state_after.get("violation_count", 0)
    components["obligation_penalty"] = -0.2 * violations

    # ── cost penalty (discount cost) ──────────────────────────────────────────
    cost_before = state_before.get("cost_to_business", 0.0)
    cost_after  = state_after.get("cost_to_business", 0.0)
    delta_cost  = max(0.0, cost_after - cost_before)
    components["cost_penalty"] = -delta_cost

    # ── stage progress bonus ──────────────────────────────────────────────────
    # reward the agent for advancing the conversation stage
    stage_order = [
        "GREETING", "DISCOVERY", "QUALIFICATION",
        "OBJECTION_HANDLING", "NEGOTIATION", "CLOSING", "POST_SALE",
    ]
    stage_before = state_before.get("stage", "GREETING")
    stage_after  = state_after.get("stage", "GREETING")
    try:
        idx_before = stage_order.index(stage_before)
        idx_after  = stage_order.index(stage_after)
        if idx_after > idx_before:
            components["stage_progress"] = 0.05 * (idx_after - idx_before)
    except ValueError:
        pass  # ESCALATED / ENDED not in list — no bonus

    # ── delay penalty ─────────────────────────────────────────────────────────
    if action.action_type == "DELAY_RESPONSE":
        components["delay_penalty"] = -0.30

    # ── terminal bonus ────────────────────────────────────────────────────────
    # outcome strings match models.py OutcomeType exactly
    if done:
        outcome          = state_after.get("outcome", "IN_PROGRESS")
        final_sat        = state_after.get("satisfaction", 0.5)
        final_ann        = state_after.get("annoyance", 0.0)
        total_cost       = state_after.get("cost_to_business", 0.0)

        if outcome == "SALE":
            bonus = 2.0 * final_sat - 1.0 * final_ann - 0.01 * total_cost
            components["terminal"] = max(bonus, 0.5)   # floor: sale always positive
        elif outcome == "ABANDONED":
            components["terminal"] = -1.5 - final_ann
        elif outcome == "NO_SALE":
            components["terminal"] = -0.5
        elif outcome == "ESCALATED":
            components["terminal"] = 0.0              # neutral — escalation not failure
        # IN_PROGRESS: no terminal bonus

    total_reward = sum(components.values())
    return float(total_reward), components