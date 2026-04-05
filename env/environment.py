"""
WhatsAppEnv – RL environment for WhatsApp sales agent training.

Dev A ownership: A1 – A5
  A1  Core models & env wiring
  A2  Step pipeline and episode control
  A3  State dynamics & obligations
  A4  Config & task loading  (constants + TaskConfig)
  A5  Testing & robustness helpers (seeding; tests live in tests/)
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from models import (
    Action,
    ActionType,
    InternalObligation,
    ObligationSummary,
    Observation,
    OutcomeType,
    State,
    StageType,
)


# ─────────────────────────────────────────────────────────────────────────────
# A4 · CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────


class C:  # namespace for global constants
    # patience / trust decay per step
    PATIENCE_DECAY: float = 0.05
    TRUST_DECAY_ON_DELAY: float = 0.10

    # obligation penalty magnitudes
    OBLIG_ANNOYANCE_PENALTY: float = 0.15
    OBLIG_TRUST_PENALTY: float = 0.10
    OBLIG_SATISFACTION_PENALTY: float = 0.10

    # conversion thresholds
    CONVERSION_THRESHOLD: float = 0.85
    PATIENCE_THRESHOLD: float = 0.15

    # default obligation deadline (steps after creation)
    DEFAULT_OBLIGATION_WINDOW: int = 4

    # noise std added to sentiment in observation
    OBSERVATION_NOISE_STD: float = 0.05

    # stage-to-conversion boosts
    STAGE_CONV_BONUS: Dict[str, float] = {
        "QUALIFICATION": 0.05,
        "NEGOTIATION": 0.08,
        "CLOSING": 0.12,
        "POST_SALE": 0.15,
    }


# ─────────────────────────────────────────────────────────────────────────────
# A4 · TASK CONFIG
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TaskConfig:
    """
    All tunable parameters for a single task difficulty.

    Produced by Dev B; consumed by WhatsAppEnv.__init__.
    Default values make it usable stand-alone.
    """

    task_id: str = "medium"
    max_steps: int = 20

    # initial state ranges  (sampled uniformly between lo and hi)
    trust_range: Tuple[float, float] = (0.4, 0.6)
    patience_range: Tuple[float, float] = (0.6, 0.9)
    satisfaction_range: Tuple[float, float] = (0.4, 0.6)
    conversion_prob_range: Tuple[float, float] = (0.3, 0.5)

    # user type distribution (weights for random.choices)
    user_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "IMPULSIVE": 0.25,
        "ANALYTICAL": 0.25,
        "SKEPTICAL": 0.15,
        "LOYAL": 0.10,
        "PRICE_SENSITIVE": 0.25,
    })

    # reward weight overrides forwarded to Dev B's reward module
    reward_weights: Dict[str, float] = field(default_factory=dict)


# Pre-built difficulty presets
TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        max_steps=15,
        trust_range=(0.5, 0.7),
        patience_range=(0.75, 0.95),
        conversion_prob_range=(0.4, 0.6),
    ),
    "medium": TaskConfig(
        task_id="medium",
        max_steps=20,
    ),
    "hard": TaskConfig(
        task_id="hard",
        max_steps=25,
        trust_range=(0.2, 0.45),
        patience_range=(0.4, 0.65),
        conversion_prob_range=(0.15, 0.35),
        user_type_weights={
            "IMPULSIVE": 0.10,
            "ANALYTICAL": 0.30,
            "SKEPTICAL": 0.35,
            "LOYAL": 0.05,
            "PRICE_SENSITIVE": 0.20,
        },
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# A2 · USER SIMULATOR PROTOCOL  (injected dependency → Dev C)
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class UserSimulatorProtocol(Protocol):
    """
    Contract that Dev C's simulator must satisfy.

    Parameters
    ----------
    action      : the agent's last Action
    state       : current ground-truth State
    rng         : seeded Random instance for reproducibility

    Returns
    -------
    (user_message: str, user_event: str)
    """

    def __call__(
        self,
        action: Action,
        state: State,
        rng: random.Random,
    ) -> Tuple[str, str]:
        ...


class _DefaultUserSimulator:
    """
    Minimal built-in simulator used when Dev C's module is unavailable.
    Covers all ActionTypes so the env never silently drops a response.
    """

    _RESPONSES: Dict[str, Tuple[str, str]] = {
        "ASK_QUESTION":     ("Let me think about that…",  "neutral"),
        "GIVE_PRICE":       ("That seems expensive.",     "skeptical"),
        "OFFER_DISCOUNT":   ("That sounds good!",          "positive"),
        "PROVIDE_INFO":     ("Thanks, that helps.",        "neutral"),
        "ESCALATE":         ("Okay, I'll wait.",           "neutral"),
        "DELAY_RESPONSE":   ("Why is this taking so long?","frustrated"),
        "END_CONVERSATION": ("Goodbye.",                   "neutral"),
    }

    # Phrases that trigger obligation creation
    FOLLOW_UP_TRIGGERS = [
        "remind me", "i'll tell you later", "let me check",
        "get back to you", "think about it",
    ]

    def __call__(
        self,
        action: Action,
        state: State,
        rng: random.Random,
    ) -> Tuple[str, str]:
        msg, event = self._RESPONSES.get(action.action_type, ("Hmm.", "neutral"))

        # PRICE_SENSITIVE users react more negatively to non-discounted prices
        if state.user_type == "PRICE_SENSITIVE" and action.action_type == "GIVE_PRICE":
            msg = "That's too pricey for me."
            event = "frustrated"

        # IMPULSIVE users can convert faster on discount
        if state.user_type == "IMPULSIVE" and action.action_type == "OFFER_DISCOUNT":
            msg = "Deal! Let's do it."
            event = "very_positive"

        # Occasionally inject a follow-up trigger
        if rng.random() < 0.15:
            msg += " Remind me tomorrow."
            event = event  # keep underlying event; obligation added separately

        return msg, event


# ─────────────────────────────────────────────────────────────────────────────
# A3 · STAGE TRANSITION TABLE
# ─────────────────────────────────────────────────────────────────────────────

# Maps (current_stage, user_event) → next_stage
# Falls back to current stage if not in table.
_STAGE_TRANSITIONS: Dict[Tuple[str, str], StageType] = {
    # GREETING → DISCOVERY when user shows interest
    ("GREETING",           "neutral"):    "DISCOVERY",
    ("GREETING",           "positive"):   "DISCOVERY",
    ("GREETING",           "skeptical"):  "OBJECTION_HANDLING",

    # DISCOVERY → QUALIFICATION when agent asks qualifying questions
    ("DISCOVERY",          "neutral"):    "QUALIFICATION",
    ("DISCOVERY",          "positive"):   "QUALIFICATION",
    ("DISCOVERY",          "skeptical"):  "OBJECTION_HANDLING",

    # QUALIFICATION → NEGOTIATION
    ("QUALIFICATION",      "neutral"):    "NEGOTIATION",
    ("QUALIFICATION",      "positive"):   "NEGOTIATION",
    ("QUALIFICATION",      "skeptical"):  "OBJECTION_HANDLING",

    # OBJECTION_HANDLING → back to NEGOTIATION once resolved
    ("OBJECTION_HANDLING", "positive"):   "NEGOTIATION",
    ("OBJECTION_HANDLING", "neutral"):    "OBJECTION_HANDLING",
    ("OBJECTION_HANDLING", "skeptical"):  "OBJECTION_HANDLING",

    # NEGOTIATION → CLOSING when very positive signal
    ("NEGOTIATION",        "very_positive"): "CLOSING",
    ("NEGOTIATION",        "positive"):      "CLOSING",
    ("NEGOTIATION",        "neutral"):       "NEGOTIATION",
    ("NEGOTIATION",        "skeptical"):     "OBJECTION_HANDLING",
    ("NEGOTIATION",        "frustrated"):    "OBJECTION_HANDLING",

    # CLOSING → POST_SALE
    ("CLOSING",            "very_positive"): "POST_SALE",
    ("CLOSING",            "positive"):      "POST_SALE",
    ("CLOSING",            "neutral"):       "CLOSING",

    # POST_SALE stays
    ("POST_SALE",          "positive"):      "POST_SALE",
    ("POST_SALE",          "very_positive"): "POST_SALE",
}

# Action types that force a stage forward (independent of user_event)
_ACTION_STAGE_PUSH: Dict[str, StageType] = {
    "ESCALATE":         "ESCALATED",
    "END_CONVERSATION": "ENDED",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class WhatsAppEnv:
    """
    WhatsApp sales-agent RL environment.

    Parameters
    ----------
    task_id   : "easy" | "medium" | "hard" or any key in TASK_CONFIGS
    config    : optional TaskConfig; overrides task_id lookup when supplied
    simulator : optional UserSimulatorProtocol; defaults to _DefaultUserSimulator
    """

    def __init__(
        self,
        task_id: str = "medium",
        config: Optional[TaskConfig] = None,
        simulator: Optional[UserSimulatorProtocol] = None,
    ):
        self.task_id = task_id
        self.config: TaskConfig = config or TASK_CONFIGS.get(task_id, TaskConfig(task_id=task_id))
        self.max_steps: int = self.config.max_steps
        self._simulator: UserSimulatorProtocol = simulator or _DefaultUserSimulator()
        self._state: Optional[State] = None
        self._chat_history: List[str] = []
        self._rng = random.Random()

    # ── A5 · seeding ──────────────────────────────────────────────────────────

    def seed(self, value: int) -> None:
        """Seed the internal RNG for fully reproducible episodes."""
        self._rng.seed(value)

    # ── A1 · public API ───────────────────────────────────────────────────────

    def state(self) -> State:
        """Return the current ground-truth State (not observable by agent)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ── A2 · reset ────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode. Samples initial state from TaskConfig ranges."""
        self._chat_history = []
        cfg = self.config

        user_types = list(cfg.user_type_weights.keys())
        weights = list(cfg.user_type_weights.values())
        user_type = self._rng.choices(user_types, weights=weights, k=1)[0]

        def uniform(lo: float, hi: float) -> float:
            return self._rng.uniform(lo, hi)

        self._state = State(
            user_type=user_type,
            true_intent="PURCHASE",
            trust=uniform(*cfg.trust_range),
            patience=uniform(*cfg.patience_range),
            satisfaction=uniform(*cfg.satisfaction_range),
            annoyance=0.0,
            conversion_prob=uniform(*cfg.conversion_prob_range),
            cost_to_business=0.0,
            stage="GREETING",
            obligations=ObligationSummary(),
            time_step=0,
            outcome="IN_PROGRESS",
            episode_done=False,
        )

        return self._build_observation()

    # ── A2 · step ─────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Advance the environment by one step."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state.episode_done:
            raise RuntimeError("Episode already done – call reset().")

        state_before = self._state.model_copy(deep=True)

        self._apply_agent_action_to_state(action)
        self._advance_time()

        user_msg, user_event = self._simulator(action, self._state, self._rng)
        self._update_state_from_user(user_msg, user_event)

        self._transition_stage(action, user_event)
        self._update_conversion_prob()

        obligation_events = self._update_obligations(action, user_msg, user_event)

        done = self._check_done()
        reward, components = self._compute_reward(state_before, action, user_event, done)

        obs = self._build_observation()
        s = self._state

        info: Dict[str, Any] = {
            "outcome": s.outcome,
            "time_step": s.time_step,
            "conversion_prob": s.conversion_prob,
            "violation_count": s.obligations.violation_count,
            "state_snapshot": {
                "outcome": s.outcome,
                "satisfaction": s.satisfaction,
                "annoyance": s.annoyance,
                "cost_to_business": s.cost_to_business,
                "trust": s.trust,
                "patience": s.patience,
                "conversion_prob": s.conversion_prob,
                "stage": s.stage,
                "time_step": s.time_step,
                "violation_count": s.obligations.violation_count,
            },
            "obligation_events": obligation_events,
            "reward_components": components,
        }

        return obs, reward, done, info

    # ─────────────────────────────────────────────────────────────────────────
    # A2 · INTERNAL PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_action(self, action):
        """Convert dict action to Action-like object if needed."""
        if hasattr(action, 'action_type'):
            return action  # already an Action object
        return type('Action', (), {
            'action_type': action.get('action_type', 'UNKNOWN'),
            'message': action.get('message', ''),
            'discount_pct': action.get('discount_pct', 0.0)
        })()

    def _apply_agent_action_to_state(self, action: Action) -> None:
        action = self._safe_action(action)
        self._chat_history.append(f"AGENT: {action.message or action.action_type}")

        s = self._state          # ← THIS LINE was the missing piece
        updates: Dict[str, Any] = {}

        if action.action_type == "OFFER_DISCOUNT":
            pct = action.discount_pct or 0.0
            updates["conversion_prob"] = s.conversion_prob + 0.10 + pct * 0.002
            updates["cost_to_business"] = s.cost_to_business + pct
            updates["satisfaction"] = s.satisfaction + 0.05

        elif action.action_type == "DELAY_RESPONSE":
            updates["trust"] = s.trust - C.TRUST_DECAY_ON_DELAY
            updates["annoyance"] = s.annoyance + 0.20
            updates["conversion_prob"] = s.conversion_prob - 0.10

        elif action.action_type == "GIVE_PRICE":
            updates["conversion_prob"] = s.conversion_prob + 0.03

        elif action.action_type == "ASK_QUESTION":
            updates["trust"] = s.trust + 0.03
            updates["satisfaction"] = s.satisfaction + 0.02

        elif action.action_type == "PROVIDE_INFO":
            updates["trust"] = s.trust + 0.05
            updates["satisfaction"] = s.satisfaction + 0.03

        elif action.action_type == "ESCALATE":
            updates["annoyance"] = s.annoyance - 0.10
            updates["trust"] = s.trust + 0.05

        elif action.action_type == "END_CONVERSATION":
            pass  # terminal action; handled in _check_done

        if action.message:
            self._maybe_create_agent_commitment(action)

        if updates:
            self._state = s.with_updates(**updates)


    def _maybe_create_agent_commitment(self, action: Action) -> None:
        """
        If the agent's message contains a commitment phrase, create an
        agent_commitment obligation.
        """
        commitment_triggers = [
            "i'll send", "i will send", "i'll check", "i will check",
            "i'll follow up", "i will follow up", "i'll get back",
        ]
        msg_lower = action.message.lower()
        if any(t in msg_lower for t in commitment_triggers):
            s = self._state
            obl = InternalObligation(
                type="agent_commitment",
                description=f"Agent committed: '{action.message[:60]}'",
                importance=0.7,
                related_stage=s.stage,
                created_at_step=s.time_step,
                due_at=s.time_step + C.DEFAULT_OBLIGATION_WINDOW,
            )
            self._state = s.with_updates(
                obligations=s.obligations.add(obl)
            )

    def _advance_time(self) -> None:
        """Increment time_step, apply patience decay, check obligation expiry."""
        s = self._state
        self._state = s.with_updates(
            time_step=s.time_step + 1,
            patience=s.patience - C.PATIENCE_DECAY,
        )
        # expiry check happens after time advances
        self._expire_overdue_obligations()

    def _expire_overdue_obligations(self) -> None:
        """Mark overdue PENDING obligations as EXPIRED and apply state penalties."""
        s = self._state
        new_obs = s.obligations

        penalty_annoyance = 0.0
        penalty_trust = 0.0
        penalty_satisfaction = 0.0

        for obl in new_obs.obligations:
            if obl.is_overdue(s.time_step):
                new_obs = new_obs.update_status(obl.obligation_id, "EXPIRED")
                penalty_annoyance += C.OBLIG_ANNOYANCE_PENALTY * obl.importance
                penalty_trust += C.OBLIG_TRUST_PENALTY * obl.importance
                penalty_satisfaction += C.OBLIG_SATISFACTION_PENALTY * obl.importance

        if penalty_annoyance > 0:
            self._state = self._state.with_updates(
                obligations=new_obs,
                annoyance=s.annoyance + penalty_annoyance,
                trust=s.trust - penalty_trust,
                satisfaction=s.satisfaction - penalty_satisfaction,
            )
        else:
            self._state = self._state.with_updates(obligations=new_obs)

    def _update_state_from_user(self, msg: str, event: str) -> None:
        """Apply state changes driven by the simulated user response."""
        self._chat_history.append(f"USER: {msg}")
        s = self._state

        deltas: Dict[str, float] = {
            "very_positive": {"satisfaction": +0.15, "conversion_prob": +0.15, "annoyance": -0.05},
            "positive":      {"satisfaction": +0.10, "conversion_prob": +0.10},
            "neutral":       {},
            "skeptical":     {"trust": -0.05, "conversion_prob": -0.05, "annoyance": +0.05},
            "frustrated":    {"annoyance": +0.20, "patience": -0.10, "conversion_prob": -0.10},
        }.get(event, {})

        if deltas:
            self._state = s.with_updates(**deltas)

        # detect follow-up promises in user message
        self._maybe_create_follow_up_obligation(msg)

    def _maybe_create_follow_up_obligation(self, user_msg: str) -> None:
        """
        If the user message contains a follow-up trigger phrase, create a
        follow_up obligation so the agent is reminded to check in.
        """
        triggers = _DefaultUserSimulator.FOLLOW_UP_TRIGGERS
        msg_lower = user_msg.lower()
        if any(t in msg_lower for t in triggers):
            s = self._state
            obl = InternalObligation(
                type="follow_up",
                description=f"User requested follow-up: '{user_msg[:60]}'",
                importance=0.5,
                related_stage=s.stage,
                created_at_step=s.time_step,
                due_at=s.time_step + C.DEFAULT_OBLIGATION_WINDOW,
            )
            self._state = s.with_updates(
                obligations=s.obligations.add(obl)
            )

    # ── A3 · stage transitions ───────────────────────────────────────────────

    def _transition_stage(self, action: Action, user_event: str) -> None:
        """
        Advance conversation stage based on action type and user event.
        Certain actions (ESCALATE, END_CONVERSATION) force specific stages.
        """
        s = self._state
        current = s.stage

        # forced transitions from action type
        if action.action_type in _ACTION_STAGE_PUSH:
            forced = _ACTION_STAGE_PUSH[action.action_type]
            if forced != current:
                self._state = s.with_updates(stage=forced)
            return

        # table-driven transition
        next_stage: Optional[StageType] = _STAGE_TRANSITIONS.get(
            (current, user_event), None
        )
        if next_stage and next_stage != current:
            self._state = self._state.with_updates(stage=next_stage)

    # ── A3 · conversion update ───────────────────────────────────────────────

    def _update_conversion_prob(self) -> None:
        """
        Recalculate conversion_prob incorporating trust, stage, and annoyance.
        Applied after stage transition so it reflects the new stage.
        """
        s = self._state

        # stage bonus
        stage_bonus = C.STAGE_CONV_BONUS.get(s.stage, 0.0)

        # trust contribution: high trust → slight boost
        trust_factor = (s.trust - 0.5) * 0.10

        # annoyance drag
        annoyance_drag = s.annoyance * 0.10

        delta = stage_bonus * 0.05 + trust_factor - annoyance_drag
        self._state = s.with_updates(conversion_prob=s.conversion_prob + delta)

    # ── A2 · obligation events ───────────────────────────────────────────────

    def _update_obligations(
        self,
        action: Action,
        user_msg: str,
        user_event: str,
    ) -> List[Dict[str, Any]]:
        """
        Emit obligation_events for this step:
          - "created"   : new obligations added this step
          - "completed" : pending obligations fulfilled by agent follow-up
          - "expired"   : obligations that expired (already handled in _advance_time)

        Returns a list of event dicts.
        """
        events: List[Dict[str, Any]] = []
        s = self._state

        # detect newly-created obligations (created_at_step == current time_step)
        for obl in s.obligations.obligations:
            if obl.created_at_step == s.time_step:
                events.append({
                    "type": "created",
                    "id": obl.obligation_id,
                    "obligation_type": obl.type,
                    "importance": obl.importance,
                    "due_at": obl.due_at,
                })

        # detect fulfilled: agent ASK_QUESTION / PROVIDE_INFO fulfils follow_up obligations
        if action.action_type in {"ASK_QUESTION", "PROVIDE_INFO", "ESCALATE"}:
            new_obs = s.obligations
            for obl in s.obligations.pending:
                if obl.type == "follow_up":
                    new_obs = new_obs.update_status(
                        obl.obligation_id, "FULFILLED", fulfilled_at=s.time_step
                    )
                    events.append({
                        "type": "completed",
                        "id": obl.obligation_id,
                        "obligation_type": obl.type,
                        "importance": obl.importance,
                    })
            if new_obs is not s.obligations:
                self._state = self._state.with_updates(obligations=new_obs)

        # report expired obligations (already marked in _advance_time)
        for obl in s.obligations.violated:
            # avoid re-reporting across steps by checking fulfilled_at
            if obl.fulfilled_at_step is None:
                events.append({
                    "type": "expired",
                    "id": obl.obligation_id,
                    "obligation_type": obl.type,
                    "importance": obl.importance,
                })

        return events

    # ── A2 · reward ───────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        before: State,
        action: Action,
        user_event: str,
        done: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Delegate to reward module; fall back to heuristic if import fails."""
        try:
            from reward.core import compute_step_reward
        except ImportError:
            import warnings
            warnings.warn(
                "reward.core not found — using fallback reward. "
                "Check that the reward/ package is on sys.path.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._fallback_reward(before, action, done)

        state_before_dict = {
            "stage":            before.stage,
            "annoyance":        before.annoyance,
            "satisfaction":     before.satisfaction,
            "cost_to_business": before.cost_to_business,
            "outcome":          before.outcome,
            "violation_count":  before.obligations.violation_count,
        }
        state_after_dict = {
            "stage":            self._state.stage,
            "annoyance":        self._state.annoyance,
            "satisfaction":     self._state.satisfaction,
            "cost_to_business": self._state.cost_to_business,
            "outcome":          self._state.outcome,
            "violation_count":  self._state.obligations.violation_count,
        }

        reward, components = compute_step_reward(
            state_before=state_before_dict,
            state_after=state_after_dict,
            action=action,
            user_event=user_event,
            done=done,
            reward_weights=self.config.reward_weights,
        )
        return reward, components
        
    def _fallback_reward(
        self,
        before: State,
        action: Action,
        done: bool,
    ) -> Tuple[float, Dict[str, float]]:
        s = self._state

        satisfaction_gain = s.satisfaction - before.satisfaction
        annoyance_gain = s.annoyance - before.annoyance
        obligation_penalty = -0.2 * s.obligations.violation_count

        terminal = 0.0
        if done:
            if s.outcome == "SALE":
                terminal = +2.0 - 0.01 * s.cost_to_business
            elif s.outcome == "ABANDONED":
                terminal = -1.5
            elif s.outcome == "NO_SALE":
                terminal = -0.5

        delay_penalty = -0.3 if action.action_type == "DELAY_RESPONSE" else 0.0

        total = (
            satisfaction_gain
            - annoyance_gain
            + obligation_penalty
            + terminal
            + delay_penalty
        )

        components = {
            "satisfaction_gain": satisfaction_gain,
            "annoyance_gain": -annoyance_gain,
            "obligation_penalty": obligation_penalty,
            "terminal": terminal,
            "delay_penalty": delay_penalty,
        }
        return total, components

    # ── A2 · done check ───────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        """
        Terminate on:
          1. conversion_prob >= CONVERSION_THRESHOLD  → SALE
          2. patience <= PATIENCE_THRESHOLD            → ABANDONED
          3. time_step >= max_steps                    → NO_SALE
          4. agent chose END_CONVERSATION              → NO_SALE
          5. agent chose ESCALATE                      → ESCALATED
        outcome is set once and frozen.
        """
        s = self._state

        if s.stage == "ENDED":
            self._state = s.with_updates(outcome="NO_SALE", episode_done=True)
            return True

        if s.stage == "ESCALATED":
            self._state = s.with_updates(outcome="ESCALATED", episode_done=True)
            return True

        if s.conversion_prob >= C.CONVERSION_THRESHOLD:
            self._state = s.with_updates(outcome="SALE", episode_done=True)
            return True

        if s.patience <= C.PATIENCE_THRESHOLD:
            self._state = s.with_updates(outcome="ABANDONED", episode_done=True)
            return True

        if s.time_step >= self.max_steps:
            self._state = s.with_updates(outcome="NO_SALE", episode_done=True)
            return True

        return False

    # ── A2 · observation builder ──────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        """
        Map internal State → agent-visible Observation.
        Adds small Gaussian noise to sentiment to model imperfect perception.
        """
        s = self._state

        # noisy sentiment: true sentiment = (satisfaction - 0.5) * 2
        true_sentiment = (s.satisfaction - 0.5) * 2.0
        noise = self._rng.gauss(0.0, C.OBSERVATION_NOISE_STD)
        noisy_sentiment = max(-1.0, min(1.0, true_sentiment + noise))

        uncertainties: List[str] = []
        if s.trust < 0.3:
            uncertainties.append("low_trust")
        if s.patience < 0.3:
            uncertainties.append("low_patience")
        if s.annoyance > 0.7:
            uncertainties.append("high_annoyance")
        if s.obligations.violation_count > 0:
            uncertainties.append(f"{s.obligations.violation_count}_obligation_violations")

        return Observation(
            chat_history=list(self._chat_history),
            stage=s.stage,
            intent="PURCHASE" if s.true_intent == "PURCHASE" else "INQUIRY",
            sentiment=noisy_sentiment,
            uncertainties=uncertainties,
            obligations=s.obligations,
            step_count=s.time_step,
        )