---
title: WhatsApp Sales RL
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - sales
  - conversational-ai
---

# WhatsApp Business Conversation RL Environment

An **OpenEnv-compliant** reinforcement learning environment that simulates real-world WhatsApp B2C sales conversations. An agent learns to optimise business outcomes — conversion rate, customer satisfaction, and cost management — through sequential decision-making across multi-turn dialogues.

---

## Why This Environment?

WhatsApp is the primary B2C sales channel for hundreds of millions of businesses globally. Every conversation is a sequential decision problem: the agent must choose when to ask questions, when to present price, when to offer a discount, and when to escalate — all while managing a customer whose trust, patience, and annoyance shift in real time.

Getting it wrong means a lost sale or a damaged relationship. Getting it right requires exactly the kind of long-horizon reasoning that RL agents are built for.

This environment models that problem faithfully:

- **Stochastic customer personas** across 5 archetypes
- **9-stage conversation funnel** (GREETING → POST_SALE)
- **Trackable obligation system** — promises made and broken have consequences
- **Dense shaped reward** providing signal on every step, not just at conversion
- **Three difficulty levels** covering the full spectrum of customer personas

---

## Environment Overview

| Property | Value |
|---|---|
| Interface | OpenEnv v1.0 (`reset` / `step` / `state`) |
| Action space | Discrete — 7 action types |
| Observation space | Structured Pydantic object |
| Episode length | 15 steps (easy) · 20 steps (medium) · 25 steps (hard) |
| Termination | Sale, abandonment, step limit, escalation, or explicit end |
| Reward | Dense shaped reward + terminal bonus |
| Stochasticity | Customer persona sampled at reset; responses stochastic |
| Server | FastAPI on port 7860 |
| Demo | Gradio UI on port 7861 |

---

## Project Structure

```
.
├── env/
│   ├── __init__.py              # make_env() factory
│   ├── environment.py           # WhatsAppEnv: reset / step / state
│   └── simulator/
│       └── user_simulator.py    # UserSimulator: stochastic customer responses
├── models.py                    # Pydantic models: Action, Observation, State, ObligationSummary
├── reward/
│   ├── __init__.py
│   ├── core.py                  # compute_step_reward() — dense per-step reward
│   └── grading.py               # grade_trajectory() — scores 0.0–1.0 per task
├── tasks/
│   ├── __init__.py
│   └── configs.py               # TaskConfig definitions for task1 / task2 / task3
├── agents/
│   ├── __init__.py
│   └── agents.py                # random_agent, rule_agent, heuristic_agent
├── app/
│   └── gradio_demo.py           # Interactive Gradio demo (stateful multi-turn)
├── server/
│   └── app.py                   # FastAPI OpenEnv server
├── inference.py                 # Baseline inference script (OpenAI client) ← root
├── openenv.yaml                 # OpenEnv task metadata
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-service compose (api + gradio)
├── smoke_test.py                # Quick environment sanity check
├── verify_configs.py            # Config consistency assertions
└── README.md                    # This file
```

---

## Action Space

The agent selects one action per step from 7 discrete types.

| Action Type | Required Fields | Effect on Hidden State |
|---|---|---|
| `ASK_QUESTION` | `message` | +0.03 trust, +0.02 satisfaction |
| `GIVE_PRICE` | `message` | +0.03 conversion probability |
| `OFFER_DISCOUNT` | `message`, `discount_pct` (0–100) | +0.10 conversion prob + 0.002 × pct, +0.05 satisfaction, cost increases |
| `PROVIDE_INFO` | `message` | +0.05 trust, +0.03 satisfaction |
| `ESCALATE` | `message` | −0.10 annoyance, +0.05 trust; terminates as ESCALATED |
| `DELAY_RESPONSE` | `message` | −0.10 trust, +0.20 annoyance, −0.10 conversion prob, −0.30 step reward |
| `END_CONVERSATION` | `message` | Terminates immediately as NO_SALE |

`OFFER_DISCOUNT` requires `discount_pct`. This is enforced at construction time by Pydantic's model validator.

**Constructing actions (Python):**

```python
from models import Action

action = Action(action_type="ASK_QUESTION", message="What is your budget?")
action = Action(action_type="OFFER_DISCOUNT", message="Here's a deal", discount_pct=15.0)
```

---

## Observation Space

Each call to `reset()` and `step()` returns an `Observation` Pydantic object.

| Field | Type | Description |
|---|---|---|
| `chat_history` | `List[str]` | Full conversation so far, alternating `AGENT:` and `USER:` prefixes |
| `stage` | `str` | Current conversation stage |
| `intent` | `str` | Inferred user intent: `PURCHASE` or `INQUIRY` |
| `sentiment` | `float` | Noisy estimate of user sentiment in [−1.0, +1.0]. Gaussian noise σ=0.05 added |
| `uncertainties` | `List[str]` | Active flags: `low_trust`, `low_patience`, `high_annoyance`, `N_obligation_violations` |
| `obligations` | `ObligationSummary` | Pending, fulfilled, and violated commitments |
| `step_count` | `int` | Number of steps taken in this episode |

**Accessing observations (Python):**

```python
obs = env.reset()

print(obs.stage)                       # "GREETING"
print(obs.sentiment)                   # e.g. 0.12
print(obs.obligations.has_pending)     # True / False
print(obs.obligations.violation_count) # int
print(obs.uncertainties)               # e.g. ["low_trust"]
```

### Conversation Stages

```
GREETING → DISCOVERY → QUALIFICATION → NEGOTIATION → CLOSING → POST_SALE
                    ↘                ↘             ↗
                  OBJECTION_HANDLING ──────────────
                                              ↓ (ESCALATE)
                                          ESCALATED
                                              ↓ (END_CONVERSATION)
                                           ENDED
```

### Obligation System

When a user says "remind me tomorrow" or "I'll get back to you", the environment creates a `follow_up` obligation. When the agent's message contains a commitment phrase ("I'll send", "I'll check"), an `agent_commitment` obligation is created. Each obligation carries a 4-step deadline.

If the agent fails to service a pending obligation before its deadline:

- +0.15 annoyance
- −0.10 trust
- −0.10 satisfaction

All multiplied by the obligation's importance weight (0–1).

---

## Hidden State (Ground Truth)

The full `State` is not visible to the agent but is returned by `env.state()` and included as `state_snapshot` in each `step()` info dict.

| Field | Type | Range | Description |
|---|---|---|---|
| `user_type` | `str` | — | `IMPULSIVE`, `ANALYTICAL`, `SKEPTICAL`, `LOYAL`, or `PRICE_SENSITIVE` |
| `true_intent` | `str` | — | Always `PURCHASE` in current tasks |
| `trust` | `float` | [0, 1] | How much the customer trusts the agent |
| `patience` | `float` | [0, 1] | Decays by 0.05 per step; episode ends if it reaches 0.15 |
| `annoyance` | `float` | [0, 1] | Accumulated friction |
| `satisfaction` | `float` | [0, 1] | Drives the noisy sentiment signal in observations |
| `conversion_prob` | `float` | [0, 1] | Episode ends as SALE when this reaches 0.85 |
| `cost_to_business` | `float` | [0, ∞) | Cumulative discount cost; penalises the terminal reward |
| `stage` | `str` | — | Current conversation stage |
| `outcome` | `str` | — | `IN_PROGRESS`, `SALE`, `NO_SALE`, `ABANDONED`, or `ESCALATED` |

---

## Reward Function

The reward is **dense** — every step produces a non-zero signal.

**Per-step components (from `reward/core.py`):**

| Component | Formula |
|---|---|
| Satisfaction gain | `satisfaction_after − satisfaction_before` |
| Annoyance penalty | `−(annoyance_after − annoyance_before)` |
| Stage progress | `max(0, stage_idx_after − stage_idx_before) × 0.05 × weight` |
| Obligation penalty | `−0.2 × violation_count` |
| Delay penalty | `−0.30` if action is `DELAY_RESPONSE`, else 0 |

**Terminal bonus (on the final step only):**

| Outcome | Bonus |
|---|---|
| `SALE` | `+2.0 − 0.01 × cost_to_business` |
| `ABANDONED` | `−1.5` |
| `NO_SALE` | `−0.5` |
| `ESCALATED` | `0.0` |

---

## Tasks and Graders

Three tasks with increasing difficulty. Each has a programmatic grader in `reward/grading.py` (`grade_trajectory`) that outputs a score in **[0.0, 1.0]**. Graders are deterministic given the same trajectory.

### Task 1 — Easy Sale (`task1`)

A warm, receptive customer already interested in the product. The agent must confirm needs, present value clearly, and close efficiently.

| Parameter | Value |
|---|---|
| `max_steps` | 15 |
| `trust` initial | [0.50, 0.70] |
| `patience` initial | [0.75, 0.95] |
| `conversion_prob` initial | [0.40, 0.60] |
| Dominant personas | IMPULSIVE (25%), PRICE_SENSITIVE (25%) |

**Grader success criteria:** Episode ends as `SALE` with fewer than 3 obligation violations. Score includes satisfaction bonus and annoyance penalty.

### Task 2 — Balanced Negotiation (`task2`)

A moderate customer with mixed signals — shows interest but raises objections and price concerns. The agent must qualify needs, handle objections, and negotiate to close.

| Parameter | Value |
|---|---|
| `max_steps` | 20 |
| `trust` initial | [0.40, 0.60] |
| `patience` initial | [0.60, 0.90] |
| `conversion_prob` initial | [0.30, 0.50] |
| Dominant personas | PRICE_SENSITIVE (25%), ANALYTICAL (25%) |

**Grader success criteria:** Episode ends as `SALE` with final `satisfaction ≥ 0.6` and `annoyance ≤ 0.4`. Failure on either threshold deducts 0.15 from the score.

### Task 3 — Hard Close (`task3`)

A skeptical, analytical customer who is hard to convince and quick to abandon. The agent must build trust patiently, resolve deep objections, and avoid missteps.

| Parameter | Value |
|---|---|
| `max_steps` | 25 |
| `trust` initial | [0.20, 0.45] |
| `patience` initial | [0.40, 0.65] |
| `conversion_prob` initial | [0.15, 0.35] |
| Dominant personas | SKEPTICAL (35%), ANALYTICAL (30%) |

**Grader success criteria:** Episode ends as `SALE` before step 20 with `trust ≥ 0.6` at termination. Late or low-trust closes are penalised.

---

## Baseline Scores

Evaluated over fixed seeded episodes (`task1=42`, `task2=43`, `task3=44`) with `temperature=0.0`.

| Agent | Task 1 Score | Task 2 Score | Task 3 Score | Mean |
|---|---|---|---|---|
| `random_agent` | 0.34 | 0.34 | 0.26 | 0.32 |
| `rule_agent` | 0.97 | 0.76 | 0.54 | 0.75 |
| `heuristic_agent` | 1.00 | 0.98 | 0.52 | 0.83 |

Scores are `grade_trajectory()` outputs in [0.0, 1.0]. A frontier LLM agent is expected to exceed the heuristic baseline on Task 2 and Task 3.

---

## Setup and Usage

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)
- A Hugging Face token or OpenAI API key (for inference)

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes (inference) | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | Yes (inference) | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes (inference) | — | Hugging Face or OpenAI API key |
| `LOCAL_IMAGE_NAME` | Optional | — | Local Docker image name if using from_docker_image() |

### Local (Python)

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/whatsapp-rl.git
cd whatsapp-rl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the OpenEnv server (port 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# 4. In a separate terminal — start the Gradio demo (port 7861)
python app/gradio_demo.py

# 5. Run the smoke test to verify the environment works end-to-end
python smoke_test.py

# 6. Run the baseline inference script
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

### Docker

```bash
# Build the image
docker build -t whatsapp-rl .

# Run the server
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="your_token_here" \
  whatsapp-rl

# Health check
curl http://localhost:7860/health
```

### Docker Compose (API + Gradio)

```bash
docker-compose up
# API:    http://localhost:7860
# Gradio: http://localhost:7861
```

---

## API Reference

The FastAPI server exposes three endpoints matching the OpenEnv spec.

### `POST /v1/reset?task_id=task1`

Start a new episode. `task_id` must be `task1`, `task2`, or `task3` (also accepts `easy`, `medium`, `hard`).

Returns the initial `Observation` as JSON.

```bash
curl -X POST "http://localhost:7860/v1/reset?task_id=task1"
```

### `POST /v1/step`

Advance one step. Body must be a valid action JSON.

```bash
curl -X POST "http://localhost:7860/v1/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "ASK_QUESTION", "message": "What is your budget?"}'

curl -X POST "http://localhost:7860/v1/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "OFFER_DISCOUNT", "discount_pct": 15.0, "message": "Special offer just for you."}'
```

Returns `StepResponse` containing observation, reward, done flag, outcome, reward components, and obligation events.

### `GET /v1/state`

Returns the full ground-truth `State`. Useful for debugging.

```bash
curl http://localhost:7860/state
```

### `GET /health`

Liveness probe. Returns server version, uptime, current task, and episode status.

```bash
curl http://localhost:7860/health
```

---

## Using the Environment Directly in Python

```python
from env import make_env
from models import Action

# Create and reset
env = make_env(task_id="task1")
obs = env.reset()

print(obs.stage)      # "GREETING"
print(obs.sentiment)  # e.g. 0.08

# Run an episode
done = False
total_reward = 0.0
trajectory = []

while not done:
    action = Action(action_type="ASK_QUESTION", message="Tell me about your needs.")
    obs, reward, done, info = env.step(action)
    trajectory.append((obs, action, reward, info))
    total_reward += reward

print(f"Outcome: {info['outcome']}  |  Total reward: {total_reward:.3f}")

# Grade the trajectory (0.0 – 1.0)
from reward.grading import grade_trajectory
score = grade_trajectory(trajectory, task_id="task1")
print(f"Grader score: {score:.3f}")

# Inspect ground truth
state = env.state()
print(f"Final trust: {state.trust:.2f}  |  Conversion prob: {state.conversion_prob:.2f}")
```

---

## Running the Baseline Agents

```python
from env import make_env
from models import Action
from agents.agents import random_agent, rule_agent, heuristic_agent
from reward.grading import grade_trajectory

env = make_env("task2")
env.seed(43)
obs = env.reset()

trajectory = []
done = False

while not done:
    action = heuristic_agent(obs)
    obs, reward, done, info = env.step(action)
    trajectory.append((obs, action, reward, info))

score = grade_trajectory(trajectory, task_id="task2")
print(f"Score: {score:.3f}")   # 0.0 – 1.0
```

---

## Inference Script

The inference script (`inference.py`) is placed in the **root directory** and runs one episode per task using an LLM agent via the OpenAI-compatible client.

**Required environment variables:**

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
```

**Run:**

```bash
python inference.py
```

**Stdout format** (strictly followed for evaluation scoring):

```
[START] task=task1 env=whatsapp_sales_rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=ASK_QUESTION reward=0.03 done=false error=null
[STEP] step=2 action=PROVIDE_INFO reward=0.05 done=false error=null
[STEP] step=3 action=GIVE_PRICE reward=0.04 done=false error=null
...
[END] success=true steps=8 score=0.87 rewards=0.03,0.05,0.04,...
```

Rules:
- One `[START]` line at episode begin
- One `[STEP]` line per step, immediately after `env.step()` returns
- One `[END]` line after episode ends — always emitted
- `reward` and `rewards` are formatted to 2 decimal places
- `done` and `success` are lowercase booleans: `true` or `false`
- `error` is the raw error string or `null`
- `score` is the `grade_trajectory()` output in [0.0, 1.0], formatted to 2 decimal places
- All fields on a single line with no newlines within a line

**Fixed seeds for reproducibility:**

| Task | Seed |
|---|---|
| task1 | 42 |
| task2 | 43 |
| task3 | 44 |

---

## Evaluation Methodology

Each task is graded by `grade_trajectory()` in `reward/grading.py`, which produces a score in [0.0, 1.0]. The grader is **deterministic** — given the same trajectory, it always returns the same score.

Grader scores are distinct from cumulative episode reward. A high reward is possible without a sale (by keeping satisfaction high for many steps), but a high grader score requires actual conversion. The grader considers:

- Whether the episode ended as a `SALE` (primary signal)
- Final satisfaction and annoyance levels
- Number of obligation violations
- Step efficiency (earlier conversion = higher score on hard tasks)
- Trust at termination (task 3 only)

---

## Pre-Submission Validation

Run the OpenEnv pre-submission validation script before submitting:

```bash
./validate-submission.sh https://your-space.hf.space .
```

This checks:
1. **HF Space is live** — pings `POST /reset` and expects HTTP 200
2. **Docker build succeeds** — runs `docker build` with a 600s timeout
3. **`openenv validate` passes** — validates `openenv.yaml` and typed models

All three must pass or the submission is disqualified.

**To verify locally before running the full script:**

```bash
# 1. Smoke test
python smoke_test.py

# 2. Config consistency
python verify_configs.py

# 3. Health check (server must be running)
curl http://localhost:7860/health

# 4. Full reset → step → state cycle
curl -X POST "http://localhost:7860/v1/reset?task_id=task1"
curl -X POST "http://localhost:7860/v1/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "ASK_QUESTION", "message": "What are you looking for?"}'
curl http://localhost:7860/v1/state
```

---

## Requirements

```
pydantic>=2.5.0
fastapi>=0.100.0
uvicorn[standard]>=0.25.0
numpy>=1.24.0
requests>=2.31.0
openai>=1.0.0
gradio>=4.0.0
python-multipart
openenv-core>=0.2.0
```

Python 3.11 or higher required.

---

## Submission Checklist

- [x] HF Space deploys and responds to `POST /reset` with HTTP 200
- [x] `openenv validate` passes — `openenv.yaml`, typed models, endpoints verified
- [x] `docker build && docker run` works end-to-end
- [x] `inference.py` in repo root — runs without error and produces `[START]`, `[STEP]`, `[END]` logs
- [x] `inference.py` uses OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [x] `[END]` line includes `score=` field in [0.0, 1.0]
- [x] 3 tasks defined (`task1`, `task2`, `task3`) with graders returning scores in [0.0, 1.0]
- [x] Graders are deterministic and reproducible
- [x] Dense reward with partial progress signals on every step
- [x] Fixed seeds per task for reproducible baseline scores
- [x] `reward/core.py` and `reward/grading.py` present and importable
- [x] Inference runtime under 20 minutes on 2 vCPU / 8 GB RAM

---
 #AUTHOR

 TEAM OBSIDIAN