from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from marketpulse import __version__
from marketpulse.adapter import build_engine, delta_available
from marketpulse.intents import ConstraintGovernor, load_intent, publish_manifest
from marketpulse.signals import CompetitiveSignal, synthetic_stream
from marketpulse.strategy import CompetitorModel, Policy, RollbackRegistry, decide

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaselineInput(StrictModel):
    competitor_price_delta: float = Field(0.0, ge=-1.0, le=1.0)
    promo_intensity: float = Field(0.0, ge=0.0, le=1.0)
    sentiment_shift: float = Field(0.0, ge=-1.0, le=1.0)
    projected_margin: float = Field(0.30, ge=0.0, le=1.0)


class SignalInput(StrictModel):
    competitor: str = Field(min_length=1, max_length=200)
    price_delta_pct: float = Field(ge=-1.0, le=1.0)
    promo_intensity: float = Field(ge=0.0, le=1.0)
    sentiment_shift: float = Field(ge=-1.0, le=1.0)
    new_launch: bool = False
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: str = Field("api", min_length=1, max_length=200)
    observed_at: str | None = Field(None, min_length=1, max_length=200)


class AnalyzeRequest(StrictModel):
    signals: list[SignalInput] = Field(min_length=1, max_length=100)
    node_ids: list[str] = Field(
        default_factory=lambda: ["marketpulse-eu-01"],
        min_length=1,
        max_length=20,
    )
    baseline: BaselineInput | None = None
    sync_nodes: bool = False
    quarantine_suspicious: bool = False


DEFAULT_BASELINE: dict[str, float] = {
    "competitor_price_delta": 0.0,
    "promo_intensity": 0.0,
    "sentiment_shift": 0.0,
    "projected_margin": 0.30,
}

app = FastAPI(
    title="MarketPulse",
    version=__version__,
    description="Intent-constrained competitive response engine powered by DeltaOS Core.",
)


@lru_cache(maxsize=1)
def _settings() -> tuple[Any, Policy]:
    intent = load_intent(CONFIG / "intent.yaml")
    policy_data = yaml.safe_load((CONFIG / "policy.yaml").read_text())
    if not isinstance(policy_data, dict):
        raise TypeError("policy file must contain a mapping")
    return intent, Policy.from_dict(policy_data)


def _validated_node_ids(node_ids: list[str]) -> list[str]:
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("node_ids must be unique")
    if any(not node_id.strip() for node_id in node_ids):
        raise ValueError("node_ids must contain non-empty values")
    return node_ids


def _coerce_signals(signals: list[SignalInput]) -> list[CompetitiveSignal]:
    return [CompetitiveSignal(**signal.model_dump()) for signal in signals]


def _run(
    signals: list[CompetitiveSignal],
    node_ids: list[str],
    baseline: dict[str, Any],
    sync_nodes: bool,
    quarantine_suspicious: bool,
) -> dict[str, Any]:
    intent, policy = _settings()
    node_ids = _validated_node_ids(node_ids)
    raw_env = {
        "segment": "mid-market-retail",
        "region": "EMEA",
        "season": "standard",
        "regulatory": ["GDPR", "UCPD"],
        "consent": True,
        "baseline": dict(baseline),
    }
    engine = build_engine(context=raw_env, intent=intent.to_payload(raw_env))
    for node_id in node_ids:
        engine.init_node(node_id, {**raw_env, "node_id": node_id})

    model = CompetitorModel(policy)
    rollback = RollbackRegistry()
    governor = ConstraintGovernor(intent)
    decisions: list[dict[str, Any]] = []
    quarantined: list[str] = []

    for index, signal in enumerate(signals):
        node_id = node_ids[index % len(node_ids)]
        input_data = signal.to_input_data(baseline)
        result = engine.run_cycle(
            input_data=input_data,
            raw_env={**raw_env, "node_id": node_id},
            node_id=node_id,
        )
        decision = decide(signal.competitor, result, policy, model=model)
        governor.observe(decision.compliance_score, decision.hard_violations)
        if decision.execution_id:
            rollback.arm(decision)
        decisions.append(decision.to_dict())

    if sync_nodes and len(node_ids) > 1:
        for source, target in pairwise(node_ids):
            engine.sync_nodes(source, target)
    if quarantine_suspicious:
        quarantined = model.quarantine_candidates()
        for competitor in quarantined:
            model.quarantine(competitor)

    return {
        "service": "marketpulse",
        "version": __version__,
        "engine_mode": engine.engine_mode,
        "delta_available": delta_available(),
        "manifest": publish_manifest(intent),
        "nodes": node_ids,
        "decisions": decisions,
        "quarantined_competitors": quarantined,
        "constraint_review": governor.recommendation(),
        "metrics": engine.metrics(),
    }


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "service": "marketpulse",
        "version": __version__,
        "status": "ready",
        "engine_mode": "live" if delta_available() else "simulation",
        "endpoints": {
            "health": "/health",
            "manifest": "/manifest",
            "analyze": "/analyze",
            "demo": "/demo",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    intent, _ = _settings()
    return {
        "status": "ok",
        "service": "marketpulse",
        "version": __version__,
        "engine_mode": "live" if delta_available() else "simulation",
        "intent_version": intent.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    intent, _ = _settings()
    return publish_manifest(intent)


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        baseline = (
            DEFAULT_BASELINE
            if request.baseline is None
            else request.baseline.model_dump()
        )
        return _run(
            signals=_coerce_signals(request.signals),
            node_ids=request.node_ids,
            baseline=baseline,
            sync_nodes=request.sync_nodes,
            quarantine_suspicious=request.quarantine_suspicious,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/demo")
def demo(
    signal_count: int = Query(8, ge=0, le=100),
    seed: int = Query(7),
) -> dict[str, Any]:
    return _run(
        signals=synthetic_stream(signal_count, seed),
        node_ids=["marketpulse-eu-01"],
        baseline=DEFAULT_BASELINE,
        sync_nodes=False,
        quarantine_suspicious=False,
    )
