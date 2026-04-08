import sys
import os
import json
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from openai import OpenAI

try:
    from env import make_env
    from models import Action
    ENV_OK = True
    print("ENV + Pydantic OK!")
except Exception as e:
    ENV_OK = False
    print(f"Import failed: {e}")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

if not API_KEY:
    print("[WARNING] No HF_TOKEN or API_KEY found — LLM calls will fail!", flush=True)

SYSTEM_PROMPT = """You are a WhatsApp sales agent. Close a SALE by building trust.

IMPORTANT: Never repeat the same action twice in a row.

Stage-based strategy:
- GREETING: use ASK_QUESTION
- DISCOVERY: use ASK_QUESTION once, then PROVIDE_INFO
- QUALIFICATION: use PROVIDE_INFO or GIVE_PRICE
- NEGOTIATION: use GIVE_PRICE or OFFER_DISCOUNT (once only, 5-15%)
- CLOSING: use PROVIDE_INFO to reinforce

Reply with JSON only — no extra text, no markdown:
{"action_type": "<ACTION>", "message": "<msg>"}

For discount:
{"action_type": "OFFER_DISCOUNT", "discount_pct": 10, "message": "<msg>"}

Actions: ASK_QUESTION, GIVE_PRICE, OFFER_DISCOUNT, PROVIDE_INFO, ESCALATE, END_CONVERSATION
"""

# rotation map so we never repeat the same action twice
_NO_REPEAT = {
    "PROVIDE_INFO":   "ASK_QUESTION",
    "ASK_QUESTION":   "PROVIDE_INFO",
    "GIVE_PRICE":     "PROVIDE_INFO",
    "OFFER_DISCOUNT": "PROVIDE_INFO",
    "ESCALATE":       "PROVIDE_INFO",
}

# stage-aware fallback when LLM is unavailable
_STAGE_FALLBACK = {
    "GREETING":           ("ASK_QUESTION",   "How can I help you today?"),
    "DISCOVERY":          ("PROVIDE_INFO",   "Let me tell you more about what we offer."),
    "QUALIFICATION":      ("GIVE_PRICE",     "Here is the pricing for our product."),
    "OBJECTION_HANDLING": ("PROVIDE_INFO",   "Let me address your concerns directly."),
    "NEGOTIATION":        ("OFFER_DISCOUNT", "I can offer you a special deal."),
    "CLOSING":            ("PROVIDE_INFO",   "You have made a great choice!"),
    "POST_SALE":          ("PROVIDE_INFO",   "Thank you for your purchase!"),
}


def _call_llm(client, obs):
    """Call the LLM and return a parsed dict. Returns None on any failure."""
    history_text = "\n".join(obs.chat_history[-6:]) if obs.chat_history else "None"

    last_action = "none"
    for line in reversed(obs.chat_history):
        if line.startswith("AGENT:"):
            last_action = line
            break

    user_prompt = (
        f"Stage: {obs.stage}\n"
        f"Sentiment: {obs.sentiment:+.2f}\n"
        f"Step: {obs.step_count}\n"
        f"Uncertainties: {', '.join(obs.uncertainties) if obs.uncertainties else 'none'}\n"
        f"Last agent action: {last_action}\n\n"
        f"Do NOT repeat the same action type as last time.\n"
        f"Last 6 conversation lines:\n{history_text}\n\n"
        f"Reply with JSON only."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        raw = (completion.choices[0].message.content or "").strip()
        print(f"[LLM RAW] {raw[:120]}", flush=True)

        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # extract first JSON object even if model adds preamble
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        return json.loads(raw)

    except Exception as exc:
        print(f"[LLM ERROR] {exc}", flush=True)
        traceback.print_exc()
        return None


def _build_action(llm_output, obs, last_action_type=""):
    """
    Convert LLM output → Action with no-repeat guard.
    Falls back to stage-aware heuristic if LLM failed.
    Returns (Action, source_label).
    """
    VALID = {
        "ASK_QUESTION", "GIVE_PRICE", "OFFER_DISCOUNT",
        "PROVIDE_INFO", "ESCALATE", "DELAY_RESPONSE", "END_CONVERSATION",
    }

    # --- LLM path ---
    if llm_output and isinstance(llm_output, dict):
        action_type = str(llm_output.get("action_type", "")).upper()
        message     = str(llm_output.get("message", ""))
        discount    = llm_output.get("discount_pct", 10.0)

        if action_type not in VALID:
            action_type = "PROVIDE_INFO"
            message = "Let me share more details."

        # block DELAY_RESPONSE
        if action_type == "DELAY_RESPONSE":
            action_type = "PROVIDE_INFO"
            message = "Let me get back to you with more details."

        # no-repeat guard
        if action_type == last_action_type and action_type in _NO_REPEAT:
            action_type = _NO_REPEAT[action_type]
            message = "Let me try a different approach."
            source = "adjusted:no_repeat"
        else:
            source = "llm"

        if action_type == "OFFER_DISCOUNT":
            pct = max(5.0, min(15.0, float(discount)))
            return Action(action_type="OFFER_DISCOUNT", message=message, discount_pct=pct), source

        return Action(action_type=action_type, message=message), source

    # --- fallback path (LLM unavailable) ---
    stage = obs.stage
    fallback_type, fallback_msg = _STAGE_FALLBACK.get(
        stage, ("PROVIDE_INFO", "Let me share more details.")
    )

    # avoid repeating even in fallback
    if fallback_type == last_action_type and fallback_type in _NO_REPEAT:
        fallback_type = _NO_REPEAT[fallback_type]
        fallback_msg  = "Here is something else that might help."

    if fallback_type == "OFFER_DISCOUNT":
        return Action(
            action_type="OFFER_DISCOUNT",
            message=fallback_msg,
            discount_pct=10.0,
        ), "fallback"

    return Action(action_type=fallback_type, message=fallback_msg), "fallback"


def _fallback_action(obs, last_action_type, reason):
    """Convenience wrapper used by tests."""
    action, source = _build_action(None, obs, last_action_type)
    return action, f"fallback:{reason}"


def run_step(task_id, history, env_state, client_state):
    if not ENV_OK:
        return history, "Environment not ready", env_state, client_state
    try:
        if not isinstance(history, list):
            history = []

        env    = env_state
        client = client_state

        if env is None or env.state().episode_done:
            clean_task = task_id.split("-")[0]
            env = make_env(clean_task)
            env.reset()

        if client is None:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        obs = env._build_observation()

        # figure out last action type from chat history
        last_action_type = ""
        for line in reversed(obs.chat_history):
            if line.startswith("AGENT:"):
                # lines are "AGENT: <message>" — we track via state below
                break

        # call LLM
        llm_output = _call_llm(client, obs)

        # build action (with no-repeat guard)
        action_obj, source = _build_action(llm_output, obs, last_action_type)
        print(f"[ACTION] {action_obj.action_type} (source={source})", flush=True)

        obs, reward, done, info = env.step(action_obj)

        # get latest customer reply
        customer_reply = "..."
        for line in reversed(obs.chat_history):
            if line.startswith("USER:"):
                customer_reply = line[len("USER: "):]
                break

        agent_message = action_obj.message or action_obj.action_type

        history = history + [
            {"role": "assistant", "content": f"Agent [{action_obj.action_type}]: {agent_message}"},
            {"role": "user",      "content": f"Customer: {customer_reply}"},
        ]

        status = (
            f"Stage: {obs.stage} | Step: {obs.step_count} | "
            f"Action: {action_obj.action_type} (src: {source})\n"
            f"Reward: {reward:+.3f} | Sentiment: {obs.sentiment:+.2f} | Done: {done}"
        )
        if done:
            status += f"\nEpisode ended — outcome: {info['outcome']}"

        return history, status, env, client

    except Exception as e:
        print(traceback.format_exc())
        return history, f"Error: {e}", env_state, client_state


def reset_env(task_id):
    if not ENV_OK:
        return [], "Environment not ready", None, None
    try:
        clean_task = task_id.split("-")[0]
        env = make_env(clean_task)
        obs = env.reset()
        s = env.state()
        status = (
            f"New episode — task: {clean_task}\n"
            f"Stage: {obs.stage} | User: {s.user_type} | "
            f"Trust: {s.trust:.2f} | Patience: {s.patience:.2f}\n"
            f"Click Agent Step to begin."
        )
        return [], status, env, None
    except Exception as e:
        print(traceback.format_exc())
        return [], f"Reset error: {e}", None, None


print("WhatsApp RL Demo starting...")
print("Open: http://localhost:7861")

with gr.Blocks(title="WhatsApp RL Agent") as demo:
    gr.Markdown("# WhatsApp Business RL Demo")
    gr.Markdown(
        "**Left = AI Sales Agent** | **Right = Simulated Customer**\n\n"
        "Click **New Episode** then **Agent Step**."
    )

    env_state    = gr.State(value=None)
    client_state = gr.State(value=None)

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["task1-easy", "task2-medium", "task3-hard"],
            value="task1-easy",
            label="Customer Difficulty",
            scale=2,
        )
        new_episode_btn = gr.Button("New Episode", variant="secondary", scale=1)

    chatbot = gr.Chatbot(value=[], height=500, label="Conversation")

    step_btn = gr.Button("Agent Step", variant="primary")

    metrics = gr.Textbox(label="Episode metrics", lines=4, interactive=False)

    step_btn.click(
        run_step,
        inputs=[task_dropdown, chatbot, env_state, client_state],
        outputs=[chatbot, metrics, env_state, client_state],
    )

    new_episode_btn.click(
        reset_env,
        inputs=[task_dropdown],
        outputs=[chatbot, metrics, env_state, client_state],
    )

demo.launch(server_name="0.0.0.0", server_port=7861, share=False)