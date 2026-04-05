# smoke_test.py — put this in your repo root and run it
from env import make_env
from models import Action

for task_id in ["task1", "task2", "task3"]:
    print(f"\n--- {task_id} ---")
    env = make_env(task_id=task_id)
    obs = env.reset()
    print(f"reset OK | stage={obs.stage} | sentiment={obs.sentiment:.2f}")

    action = Action(action_type="ASK_QUESTION", message="Tell me about your needs.")
    obs2, reward, done, info = env.step(action)
    print(f"step OK  | reward={reward:.3f} | outcome={info['outcome']}")

    state = env.state()
    print(f"state OK | trust={state.trust:.2f} | conversion={state.conversion_prob:.2f}")

    # confirm reward is coming from core.py not fallback
    components = info["reward_components"]
    print(f"reward components: {list(components.keys())}")
    