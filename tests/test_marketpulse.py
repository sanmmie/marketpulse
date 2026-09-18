import pytest

from marketpulse.adapter import build_engine, compute_delta, score_compliance
from marketpulse.audit import AuditTrail
from marketpulse.intents import ConstraintGovernor, Intent
from marketpulse.signals import CompetitiveSignal
from marketpulse.strategy import (
    CompetitorModel,
    Policy,
    ResponseTier,
    RollbackRegistry,
    decide,
)

INTENT = {
    "goal": "protect_margin",
    "rationale": "value over price",
    "constraints": {"min_margin": 0.22, "max_discount": 0.15, "no_price_war": True},
    "ethical_constraints": ["transparency"],
}


def _result(delta, compliance, violations=None, hard_violations=None, **overrides):
    result = {
        "delta": delta,
        "intent_compliance_score": compliance,
        "violations": violations or [],
        "hard_violations": hard_violations or [],
        "context_signature": "abc123",
        "signal_confidence": 1.0,
        "latency_ms": 1.0,
    }
    result.update(overrides)
    return result


def test_compliance_flags_margin_breach():
    delta = compute_delta({"projected_margin": 0.30}, {"projected_margin": 0.10})
    out = score_compliance(delta, INTENT)
    assert out["score"] < 1.0
    assert out["allowed"] is False
    assert any("min_margin" in violation for violation in out["violations"])


def test_hard_violation_forces_hold_even_with_high_score():
    policy = Policy()
    decision = decide(
        "Acme",
        _result(
            {"competitor_price_delta": -0.01},
            1.0,
            violations=["max_discount: 0.20 > 0.15"],
            hard_violations=["max_discount: 0.20 > 0.15"],
        ),
        policy,
    )
    assert decision.tier is ResponseTier.HOLD
    assert decision.hard_violations


def test_low_compliance_forces_hold():
    policy = Policy()
    decision = decide("Acme", _result({"competitor_price_delta": -0.2}, 0.4), policy)
    assert decision.tier is ResponseTier.HOLD


def test_low_signal_confidence_forces_hold():
    policy = Policy()
    decision = decide(
        "Acme",
        _result({"competitor_price_delta": -0.2}, 1.0, signal_confidence=0.5),
        policy,
    )
    assert decision.tier is ResponseTier.HOLD
    assert "signal confidence" in decision.reason


def test_unconfirmed_price_war_forces_hold():
    policy = Policy()
    decision = decide(
        "Acme",
        _result(
            {
                "competitor_price_delta": -0.04,
                "price_war_signal": 1,
                "price_war_confidence": 0.5,
            },
            1.0,
            signal_confidence=0.9,
        ),
        policy,
    )
    assert decision.tier is ResponseTier.HOLD
    assert "unconfirmed" in decision.reason


def test_latency_budget_forces_hold():
    policy = Policy()
    decision = decide(
        "Acme",
        _result({"competitor_price_delta": -0.04}, 1.0, latency_ms=501),
        policy,
    )
    assert decision.tier is ResponseTier.HOLD
    assert "latency" in decision.reason


def test_small_delta_ignored():
    policy = Policy()
    decision = decide("Acme", _result({"competitor_price_delta": -0.01}, 0.95), policy)
    assert decision.tier is ResponseTier.IGNORE


def test_large_delta_differentiates():
    policy = Policy()
    decision = decide("Acme", _result({"competitor_price_delta": -0.20}, 0.95), policy)
    assert decision.tier is ResponseTier.DIFFERENTIATE


def test_repeated_suspicious_signals_force_hold():
    policy = Policy()
    model = CompetitorModel(policy)
    decisions = []
    for _ in range(3):
        decisions.append(
            decide(
                "Acme",
                _result(
                    {
                        "competitor_price_delta": -0.01,
                        "price_war_signal": 1,
                        "price_war_confidence": 0.5,
                    },
                    1.0,
                    signal_confidence=0.9,
                ),
                policy,
                model=model,
            )
        )
    assert decisions[-1].tier is ResponseTier.HOLD
    assert "pattern" in decisions[-1].reason


def test_mirror_action_limit_forces_differentiation():
    policy = Policy()
    model = CompetitorModel(policy)
    decisions = []
    for _ in range(3):
        decisions.append(
            decide(
                "Acme",
                _result({"competitor_price_delta": -0.05}, 1.0),
                policy,
                model=model,
            )
        )
    assert [decision.tier for decision in decisions] == [
        ResponseTier.MIRROR,
        ResponseTier.MIRROR,
        ResponseTier.DIFFERENTIATE,
    ]


def test_rollback_registry_requires_armed_execution():
    policy = Policy()
    decision = decide("Acme", _result({"competitor_price_delta": -0.05}, 1.0), policy)
    registry = RollbackRegistry()
    plan = registry.arm(decision)
    assert plan is not None
    assert registry.status(decision.execution_id) == "armed"
    assert registry.rollback(decision.execution_id)["status"] == "rolled_back"
    with pytest.raises(ValueError):
        registry.rollback(decision.execution_id)


def test_authenticated_audit_detects_tampering(tmp_path):
    key = "0123456789abcdef0123456789abcdef"
    trail = AuditTrail(tmp_path / "a.jsonl", key=key)
    trail.append({"event": "x"})
    trail.append({"event": "y"})
    assert trail.authenticated
    assert trail.verify_chain()

    path = tmp_path / "a.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"x"', '"tampered"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not trail.verify_chain()


def test_authenticated_audit_rejects_short_key(tmp_path):
    with pytest.raises(ValueError):
        AuditTrail(tmp_path / "a.jsonl", key="short")


def test_signal_validation_rejects_bad_confidence():
    with pytest.raises(ValueError):
        CompetitiveSignal("Acme", -0.1, 0.2, 0.0, confidence=1.1)


def test_compute_delta_ignores_non_numeric_metadata():
    delta = compute_delta(
        {"projected_margin": 0.30, "source": "old"},
        {"projected_margin": 0.28, "source": None},
    )
    assert delta == {"projected_margin": -0.02}


def test_intent_profiles_and_governor_recommendation():
    intent = Intent(
        goal="protect margin",
        constraints={"min_margin": 0.22, "max_discount": 0.15},
        constraint_profiles={"peak": {"min_margin": 0.25}},
    )
    assert (
        intent.effective_constraints({"constraint_profile": "peak"})["min_margin"]
        == 0.25
    )
    governor = ConstraintGovernor(intent, window_size=2)
    governor.observe(0.5, ["min_margin: 0.20 < 0.22"])
    governor.observe(0.5, ["min_margin: 0.20 < 0.22"])
    recommendation = governor.recommendation()
    assert recommendation["status"] == "review_required"
    assert recommendation["changes"]["min_margin"] == 0.23


def test_engine_runs_in_simulation_when_core_absent():
    engine = build_engine(context={"baseline": {}}, intent=INTENT)
    result = engine.run_cycle(
        input_data={"proposed_discount": 0.05, "projected_margin": 0.28},
        raw_env={"baseline": {"proposed_discount": 0.0, "projected_margin": 0.30}},
    )
    assert result["engine_mode"] == "simulation"
    assert result["allowed"] is True
    assert result["latency_ms"] >= 0
    assert engine.metrics()["cycles"] == 1


def test_policy_rejects_unknown_fields():
    with pytest.raises(ValueError):
        Policy.from_dict({"min_compliance": 0.7, "unknown": True})
