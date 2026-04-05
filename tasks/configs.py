"""
tasks/configs.py

Single source of truth for task configurations lives in env/environment.py.
This module re-exports from there so any code that imports from tasks.configs
continues to work without modification.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env"))
from environment import TaskConfig, TASK_CONFIGS


def get_task_config(task_id: str) -> TaskConfig:
    """
    Get TaskConfig by task_id.

    Accepts both OpenEnv-style IDs (task1, task2, task3)
    and difficulty names (easy, medium, hard).

    Raises KeyError if task_id is not recognised.
    """
    # normalise task1/task2/task3 → easy/medium/hard
    _ALIAS = {"task1": "easy", "task2": "medium", "task3": "hard"}
    resolved = _ALIAS.get(task_id, task_id)

    if resolved not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Valid values: {list(_ALIAS.keys()) + list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[resolved]


def get_openenv_config(task_id: str) -> dict:
    """
    Returns a plain dict for OpenEnv server compatibility.
    """
    cfg = get_task_config(task_id)
    return {
        "task_id": task_id,          # keep original ID (task1/task2/task3)
        "max_steps": cfg.max_steps,
        "user_type_weights": cfg.user_type_weights,
        "trust_range": cfg.trust_range,
        "patience_range": cfg.patience_range,
        "satisfaction_range": cfg.satisfaction_range,
        "conversion_prob_range": cfg.conversion_prob_range,
        "reward_weights": cfg.reward_weights,
    }


__all__ = ["TaskConfig", "TASK_CONFIGS", "get_task_config", "get_openenv_config"]