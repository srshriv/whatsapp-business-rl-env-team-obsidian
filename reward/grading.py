"""
Trajectory grading for OpenEnv.
Input: list of (obs, action, reward, info) tuples
Output: single float score in [0, 1]
"""

from typing import List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryScore:
    total_reward: float
    outcome: str
    final_score: float
    satisfaction: float
    annoyance: float
    violation_count: int


# Base scores per outcome — drive the grade, not step rewards
_OUTCOME_BASE: dict = {
    "SALE":        0.75,
    "ESCALATED":   0.35,
    "NO_SALE":     0.20,
    "ABANDONED":   0.10,
    "IN_PROGRESS": 0.15,   # episode cut off without terminal signal
}


def grade_trajectory(
    trajectory: List[Tuple[Any, Any, float, Any]],
    task_id: Optional[str] = None,
) -> float:
    """
    Grades a complete episode trajectory.

    Args:
        trajectory: List of (obs, action, step_reward, info) tuples.
                    The final tuple's `info` must contain:
                      - "outcome"          (str)  : terminal OutcomeType
                      - "state_snapshot"   (dict) : final ground-truth state
                      - "violation_count"  (int)  : total obligation violations

    Returns:
        Final score in [0.0, 1.0].
    """
    if not trajectory:
        return 0.05

    # ── 1. Extract terminal info from the last step ───────────────────────────
    _, _, _, last_info = trajectory[-1]

    outcome         = last_info.get("outcome", "IN_PROGRESS")
    state_snapshot  = last_info.get("state_snapshot", {})
    violation_count = last_info.get("violation_count", 0)

    final_satisfaction = state_snapshot.get("satisfaction", 0.5)
    final_annoyance    = state_snapshot.get("annoyance", 0.0)
    final_trust        = state_snapshot.get("trust", 0.5)
    final_step         = last_info.get("time_step", state_snapshot.get("time_step", len(trajectory)))

    # ── 2. Base score driven entirely by outcome ──────────────────────────────
    base = _OUTCOME_BASE.get(outcome, 0.15)

    # ── 3. Modifiers (bounded so they can't flip the outcome category) ────────

    # Satisfaction nudges score up to ±0.10
    satisfaction_mod = (final_satisfaction - 0.5) * 0.20   # range [-0.10, +0.10]

    # Annoyance pushes score down up to -0.10
    annoyance_mod = -final_annoyance * 0.10                 # range [-0.10,  0.00]

    # Each obligation violation costs 0.03, capped at -0.15
    obligation_mod = -min(violation_count * 0.03, 0.15)

    # ── 4. Task-specific objective shaping ─────────────────────────────────────
    task_bonus = 0.0
    task_penalty = 0.0
    if task_id == "task1":
        if outcome == "SALE" and violation_count < 3:
            task_bonus += 0.08
        elif violation_count >= 3:
            task_penalty += 0.08
    elif task_id == "task2":
        if outcome == "SALE" and final_satisfaction >= 0.6 and final_annoyance <= 0.4:
            task_bonus += 0.10
        elif outcome == "SALE":
            task_penalty += 0.08
    elif task_id == "task3":
        if outcome == "SALE" and final_trust >= 0.6 and final_step < 20:
            task_bonus += 0.12
        elif outcome == "SALE":
            task_penalty += 0.10

    # ── 5. Combine and clamp to [0, 1] ────────────────────────────────────────
    # ── 5. Combine and clamp strictly to (0.01, 0.99) ─────────────────────────
    raw = base + satisfaction_mod + annoyance_mod + obligation_mod + task_bonus - task_penalty
    final_score = max(0.05, min(0.95, raw))

    return round(final_score, 4)

# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sale_info = {
        "outcome": "SALE",
        "state_snapshot": {"satisfaction": 0.8, "annoyance": 0.1},
        "violation_count": 0,
    }
    abandoned_info = {
        "outcome": "ABANDONED",
        "state_snapshot": {"satisfaction": 0.2, "annoyance": 0.8},
        "violation_count": 2,
    }

    sale_traj      = [({}, {}, 0.3, {})] * 9 + [({}, {}, 0.5, sale_info)]
    abandoned_traj = [({}, {}, 0.1, {})] * 9 + [({}, {}, -0.2, abandoned_info)]

    print(f"SALE score:      {grade_trajectory(sale_traj, task_id='task1')}")       # expect ~0.88
    print(f"ABANDONED score: {grade_trajectory(abandoned_traj, task_id='task3')}")  # expect ~0.01
