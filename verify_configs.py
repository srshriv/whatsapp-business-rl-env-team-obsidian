# verify_configs.py
from tasks.configs import get_task_config, get_openenv_config
from env import make_env

# both aliases must resolve to the same config
assert get_task_config("task1").max_steps == get_task_config("easy").max_steps == 15
assert get_task_config("task2").max_steps == get_task_config("medium").max_steps == 20
assert get_task_config("task3").max_steps == get_task_config("hard").max_steps == 25

# user type keys must match models.py UserType
from models import State
valid_user_types = {"IMPULSIVE", "ANALYTICAL", "SKEPTICAL", "LOYAL", "PRICE_SENSITIVE"}
for tid in ["task1", "task2", "task3"]:
    cfg = get_task_config(tid)
    assert set(cfg.user_type_weights.keys()) == valid_user_types, \
        f"{tid} has invalid user type keys: {cfg.user_type_weights.keys()}"

# make_env must work with both naming schemes
for tid in ["task1", "task2", "task3", "easy", "medium", "hard"]:
    env = make_env(tid)
    obs = env.reset()
    assert obs.stage == "GREETING"
    print(f"make_env('{tid}') OK — max_steps={env.max_steps}")

print("\nAll checks passed. Single source of truth confirmed.")