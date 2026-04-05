"""
env/__init__.py
"""
from __future__ import annotations

from .environment import WhatsAppEnv, TaskConfig, TASK_CONFIGS
from .simulator.user_simulator import default_simulator
from tasks.configs import get_task_config


def make_env(task_id: str = "task1", config: TaskConfig | None = None) -> WhatsAppEnv:
    """
    Factory function. Accepts task1/task2/task3 or easy/medium/hard.

    Examples
    --------
    env = make_env("task1")   # easy difficulty
    env = make_env("hard")    # same as task3
    """
    resolved_config = config or get_task_config(task_id)
    return WhatsAppEnv(
        task_id=task_id,
        config=resolved_config,
        simulator=default_simulator,
    )


__all__ = ["make_env", "WhatsAppEnv", "TaskConfig", "TASK_CONFIGS"]