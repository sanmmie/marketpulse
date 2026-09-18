from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ResponseTier(str, Enum):
    HOLD = "hold_for_human"
    IGNORE = "ignore"
    MIRROR = "mirror"
    DIFFERENTIATE = "differentiate"


@dataclass
class Policy:
    min_compliance: float = 0.70
    noise_threshold: float = 0.03
    mirror_threshold: float = 0.10
    min_signal_confidence: float = 0.80
    latency_budget_ms: float = 500.0
    price_war_confirmation_threshold: float = 0.80
    max_consecutive_price_war_signals: int = 2
    max_mirror_actions_per_competitor: int = 2

    def __post_init__(self) -> None:
        for name in (
            "min_compliance",
            "noise_threshold",
            "mirror_threshold",
            "min_signal_confidence",
            "latency_budget_ms",
            "price_war_confirmation_threshold",
        ):
            if isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be numeric")
            value = _finite_float(getattr(self, name), 0.0)
            setattr(self, name, value)
        for name in (
            "max_consecutive_price_war_signals",
            "max_mirror_actions_per_competitor",
        ):
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), int
            ):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.min_compliance <= 1:
            raise ValueError("min_compliance must be between 0 and 1")
        if self.noise_threshold < 0:
            raise ValueError("noise_threshold must be non-negative")
        if self.mirror_threshold <= self.noise_threshold:
            raise ValueError("mirror_threshold must exceed noise_threshold")
        if not 0 <= self.min_signal_confidence <= 1:
            raise ValueError("min_signal_confidence must be between 0 and 1")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        if not 0 <= self.price_war_confirmation_threshold <= 1:
            raise ValueError("price_war_confirmation_threshold must be between 0 and 1")
        if self.max_consecutive_price_war_signals < 1:
            raise ValueError("max_consecutive_price_war_signals must be positive")
        if self.max_mirror_actions_per_competitor < 1:
            raise ValueError("max_mirror_actions_per_competitor must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        if not isinstance(data, dict):
            raise TypeError("policy must be a mapping")
        allowed = set(cls.__annotations__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class CompetitorState:
    competitor: str
    consecutive_cuts: int = 0
    confirmed_price_war_signals: int = 0
    suspicious_signals: int = 0
    non_war_signals: int = 0
    mirror_actions: int = 0
    retaliation_risk: float = 0.0
    quarantined: bool = False

    def observe(
        self,
        delta: dict[str, Any],
        signal_confidence: float,
        price_war_signal: float,
        price_war_confidence: float,
        confirmation_threshold: float,
    ) -> None:
        if self.quarantined:
            return
        price_delta = _finite_float(delta.get("competitor_price_delta", 0.0), 0.0)
        self.consecutive_cuts = self.consecutive_cuts + 1 if price_delta < 0 else 0
        if price_war_signal > 0:
            self.non_war_signals = 0
            if price_war_confidence >= confirmation_threshold:
                self.confirmed_price_war_signals += 1
            else:
                self.suspicious_signals += 1
        else:
            self.confirmed_price_war_signals = 0
            self.non_war_signals += 1
            if self.non_war_signals >= 3:
                self.suspicious_signals = 0
        cut_pressure = min(1.0, self.consecutive_cuts / 5) * signal_confidence
        war_pressure = min(
            1.0,
            max(self.confirmed_price_war_signals, self.suspicious_signals) / 3,
        ) * max(signal_confidence, price_war_confidence)
        self.retaliation_risk = round(max(cut_pressure, war_pressure), 3)

    def quarantine(self) -> None:
        self.quarantined = True
        self.retaliation_risk = 1.0

    def should_hold(self, policy: Policy) -> bool:
        return not self.quarantined and (
            self.suspicious_signals >= 3
            or self.confirmed_price_war_signals
            >= policy.max_consecutive_price_war_signals
        )


class CompetitorModel:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.states: dict[str, CompetitorState] = {}

    def observe(
        self,
        competitor: str,
        delta: dict[str, Any],
        signal_confidence: float,
        price_war_signal: float,
        price_war_confidence: float,
    ) -> CompetitorState:
        state = self.states.setdefault(competitor, CompetitorState(competitor))
        state.observe(
            delta,
            signal_confidence,
            price_war_signal,
            price_war_confidence,
            self.policy.price_war_confirmation_threshold,
        )
        return state

    def record_tier(self, competitor: str, tier: ResponseTier) -> CompetitorState:
        state = self.states.get(competitor)
        if state is None:
            state = self.states[competitor] = CompetitorState(competitor)
        if tier is ResponseTier.MIRROR and not state.quarantined:
            state.mirror_actions += 1
        else:
            state.mirror_actions = 0
        cut_pressure = min(1.0, state.consecutive_cuts / 5)
        war_pressure = min(
            1.0,
            max(state.confirmed_price_war_signals, state.suspicious_signals) / 3,
        )
        state.retaliation_risk = round(max(cut_pressure, war_pressure), 3)
        return state

    def quarantine(self, competitor: str) -> CompetitorState:
        state = self.states.get(competitor)
        if state is None:
            state = self.states[competitor] = CompetitorState(competitor)
        state.quarantine()
        return state

    def risk_for(self, competitor: str) -> float:
        return self.states.get(competitor, CompetitorState(competitor)).retaliation_risk

    def quarantine_candidates(self) -> list[str]:
        return [
            competitor
            for competitor, state in self.states.items()
            if not state.quarantined and state.suspicious_signals >= 3
        ]


@dataclass
class Decision:
    competitor: str
    tier: ResponseTier
    delta_magnitude: float
    compliance_score: float
    violations: list[str]
    recommended_action: str
    context_signature: str
    signal_confidence: float = 1.0
    latency_ms: float = 0.0
    hard_violations: list[str] = field(default_factory=list)
    execution_id: str | None = None
    rollback_action: str | None = None
    reason: str = ""
    node_id: str | None = None
    auto_execute: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tier"] = self.tier.value
        return result


@dataclass
class RollbackPlan:
    execution_id: str
    action: str
    compensating_action: str
    status: str = "armed"


class RollbackRegistry:
    def __init__(self):
        self._plans: dict[str, RollbackPlan] = {}

    def arm(self, decision: Decision) -> RollbackPlan | None:
        if not decision.execution_id or not decision.rollback_action:
            return None
        plan = RollbackPlan(
            execution_id=decision.execution_id,
            action=decision.recommended_action,
            compensating_action=decision.rollback_action,
        )
        self._plans[decision.execution_id] = plan
        return plan

    def rollback(self, execution_id: str) -> dict[str, Any]:
        plan = self._plans.get(execution_id)
        if plan is None:
            raise KeyError(f"unknown execution_id: {execution_id}")
        if plan.status != "armed":
            raise ValueError("execution is not armed")
        plan.status = "rolled_back"
        return {
            "execution_id": execution_id,
            "status": plan.status,
            "compensating_action": plan.compensating_action,
        }

    def status(self, execution_id: str) -> str:
        plan = self._plans.get(execution_id)
        if plan is None:
            raise KeyError(f"unknown execution_id: {execution_id}")
        return plan.status


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {value!r}")
    return result


def _magnitude(delta: dict[str, Any]) -> float:
    keys = ("competitor_price_delta", "promo_intensity", "sentiment_shift")
    return round(
        max(abs(_finite_float(delta.get(key, 0.0), 0.0)) for key in keys),
        4,
    )


def _action_for(tier: ResponseTier, competitor: str, delta: dict[str, Any]) -> str:
    if tier is ResponseTier.HOLD:
        return "Escalate to pricing committee - intent constraints at risk."
    if tier is ResponseTier.IGNORE:
        return "No action. Delta within noise band; protect brand price stability."
    if tier is ResponseTier.MIRROR:
        return (
            f"Partial follow on {competitor}: match up to "
            f"{abs(_finite_float(delta.get('competitor_price_delta', 0.0), 0.0)) * 0.5:.1%} only."
        )
    return (
        f"Differentiate against {competitor}: hold price, launch value "
        "counter-narrative (bundling / loyalty / service tier)."
    )


def decide(
    signal_competitor: str,
    engine_result: dict[str, Any],
    policy: Policy,
    model: CompetitorModel | None = None,
) -> Decision:
    if not isinstance(engine_result, dict):
        raise TypeError("engine_result must be a mapping")
    delta = engine_result.get("delta", {}) or {}
    if not isinstance(delta, dict):
        raise TypeError("engine_result.delta must be a mapping")
    compliance = round(
        max(
            0.0,
            min(
                1.0,
                _finite_float(engine_result.get("intent_compliance_score", 0.0), 0.0),
            ),
        ),
        3,
    )
    violations = engine_result.get("violations", []) or []
    if not isinstance(violations, list) or not all(
        isinstance(item, str) for item in violations
    ):
        raise ValueError("engine_result.violations must be a list of strings")
    hard_violations = engine_result.get("hard_violations", []) or []
    if not isinstance(hard_violations, list) or not all(
        isinstance(item, str) for item in hard_violations
    ):
        raise ValueError("engine_result.hard_violations must be a list of strings")
    if not hard_violations:
        hard_violations = [
            violation
            for violation in violations
            if any(
                marker in violation
                for marker in (
                    "min_margin",
                    "max_discount",
                    "no_price_war",
                    "transparency",
                )
            )
        ]
    signal_confidence = round(
        max(
            0.0,
            min(1.0, _finite_float(engine_result.get("signal_confidence", 1.0), 1.0)),
        ),
        3,
    )
    latency_ms = _finite_float(engine_result.get("latency_ms", 0.0), 0.0)
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    price_war_signal = _finite_float(delta.get("price_war_signal", 0.0), 0.0)
    if price_war_signal not in (0.0, 1.0):
        raise ValueError("price_war_signal must be 0 or 1")
    price_war_confidence = round(
        max(
            0.0,
            min(
                1.0,
                _finite_float(
                    delta.get("price_war_confidence", signal_confidence),
                    signal_confidence,
                ),
            ),
        ),
        3,
    )
    state = None
    if model is not None:
        state = model.observe(
            signal_competitor,
            delta,
            signal_confidence,
            price_war_signal,
            price_war_confidence,
        )

    mag = _magnitude(delta)
    reasons: list[str] = []
    hold_reasons: list[str] = []
    tier = ResponseTier.IGNORE
    if hard_violations:
        tier = ResponseTier.HOLD
        hold_reasons.append("hard intent violation")
    if compliance < policy.min_compliance:
        tier = ResponseTier.HOLD
        hold_reasons.append("compliance below floor")
    if signal_confidence < policy.min_signal_confidence:
        tier = ResponseTier.HOLD
        hold_reasons.append("signal confidence below floor")
    if latency_ms > policy.latency_budget_ms:
        tier = ResponseTier.HOLD
        hold_reasons.append("latency budget exceeded")
    if (
        price_war_signal > 0
        and price_war_confidence < policy.price_war_confirmation_threshold
    ):
        tier = ResponseTier.HOLD
        hold_reasons.append("price-war signal unconfirmed")
    if state is not None and state.quarantined:
        tier = ResponseTier.HOLD
        hold_reasons.append("competitor is quarantined")
    if state is not None and state.should_hold(policy):
        tier = ResponseTier.HOLD
        hold_reasons.append("competitor signal pattern requires review")
    if tier is ResponseTier.HOLD:
        reasons.extend(hold_reasons)
    elif mag < policy.noise_threshold:
        tier = ResponseTier.IGNORE
        reasons.append("within noise band")
    elif mag < policy.mirror_threshold:
        tier = ResponseTier.MIRROR
        if (
            state is not None
            and state.mirror_actions >= policy.max_mirror_actions_per_competitor
        ):
            tier = ResponseTier.DIFFERENTIATE
            reasons.append("mirror action limit reached")
        else:
            reasons.append("material but mirrorable signal")
    else:
        tier = ResponseTier.DIFFERENTIATE
        reasons.append("material signal requires differentiation")

    if model is not None:
        state = model.record_tier(signal_competitor, tier)

    execution_id = None
    rollback_action = None
    auto_execute = False
    if tier is ResponseTier.MIRROR:
        execution_id = f"mp-{uuid.uuid4().hex}"
        rollback_action = "Restore the prior price and cancel the partial match."
        auto_execute = False
    elif tier is ResponseTier.DIFFERENTIATE:
        execution_id = f"mp-{uuid.uuid4().hex}"
        rollback_action = (
            "Withdraw the value counter-narrative and restore the prior campaign."
        )
        auto_execute = False

    return Decision(
        competitor=signal_competitor,
        tier=tier,
        delta_magnitude=mag,
        compliance_score=compliance,
        violations=violations,
        hard_violations=hard_violations,
        recommended_action=_action_for(tier, signal_competitor, delta),
        context_signature=str(engine_result.get("context_signature", "")),
        signal_confidence=signal_confidence,
        latency_ms=latency_ms,
        reason="; ".join(reasons),
        node_id=engine_result.get("node_id"),
        execution_id=execution_id,
        rollback_action=rollback_action,
        auto_execute=auto_execute,
    )
