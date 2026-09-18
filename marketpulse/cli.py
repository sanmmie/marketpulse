from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import yaml

from .adapter import build_engine, delta_available
from .audit import AuditTrail
from .intents import ConstraintGovernor, load_intent, publish_manifest
from .signals import CompetitiveSignal, synthetic_stream
from .strategy import CompetitorModel, Policy, RollbackRegistry, decide

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
ARTIFACTS = ROOT / "artifacts"


def _nodes(values: list[str] | None) -> list[str]:
    nodes = values or ["marketpulse-eu-01"]
    if not nodes or any(not node.strip() for node in nodes):
        raise ValueError("nodes must contain non-empty node ids")
    if len(nodes) != len(set(nodes)):
        raise ValueError("node ids must be unique")
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser(description="MarketPulse on DeltaOS Core")
    ap.add_argument("--signals", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--node", action="append", dest="nodes")
    ap.add_argument("--sync-nodes", action="store_true")
    ap.add_argument("--quarantine-suspicious", action="store_true")
    ap.add_argument("--allow-unauthenticated-audit", action="store_true")
    args = ap.parse_args()
    if args.signals < 0:
        raise ValueError("signals must be non-negative")

    intent = load_intent(CONFIG / "intent.yaml")
    policy = Policy.from_dict(yaml.safe_load((CONFIG / "policy.yaml").read_text()))
    node_ids = _nodes(args.nodes)
    raw_env = {
        "segment": "mid-market-retail",
        "region": "EMEA",
        "season": "standard",
        "regulatory": ["GDPR", "UCPD"],
        "consent": True,
        "baseline": {
            "competitor_price_delta": 0.0,
            "promo_intensity": 0.0,
            "sentiment_shift": 0.0,
            "projected_margin": 0.30,
        },
    }
    engine = build_engine(context=raw_env, intent=intent.to_payload(raw_env))
    for node_id in node_ids:
        engine.init_node(node_id, {**raw_env, "node_id": node_id})

    audit = AuditTrail(
        ARTIFACTS / "audit.jsonl",
        require_authentication=not args.allow_unauthenticated_audit,
    )
    audit.append(
        {
            "event": "engine_start",
            "mode": engine.engine_mode,
            "nodes": node_ids,
            "intent_version": intent.version,
        }
    )
    model = CompetitorModel(policy)
    rollback = RollbackRegistry()
    governor = ConstraintGovernor(intent)

    print("=" * 72)
    print("MarketPulse - Intent-Constrained Competitive Response Engine")
    print(f"DeltaOS core: {'LIVE' if delta_available() else 'SIMULATION'}")
    print("=" * 72)
    print("\n[Intent Manifest]")
    print(json.dumps(publish_manifest(intent), indent=2, sort_keys=True))

    print("\n[Cycle Log]")
    print("-" * 72)
    signals: list[CompetitiveSignal] = synthetic_stream(args.signals, args.seed)
    for index, sig in enumerate(signals):
        node_id = node_ids[index % len(node_ids)]
        input_data = sig.to_input_data(raw_env["baseline"])
        result = engine.run_cycle(
            input_data=input_data,
            raw_env={**raw_env, "node_id": node_id},
            node_id=node_id,
        )
        decision = decide(sig.competitor, result, policy, model=model)
        governor.observe(
            decision.compliance_score,
            decision.hard_violations,
        )
        if decision.execution_id:
            rollback.arm(decision)
        print(
            f"{decision.tier.value.upper():<14} "
            f"| {decision.competitor:<15} "
            f"| node={node_id:<18} "
            f"| delta={decision.delta_magnitude:.3f} "
            f"| compliance={decision.compliance_score:.3f} "
            f"| confidence={decision.signal_confidence:.3f} "
            f"| latency={decision.latency_ms:.3f}ms"
        )
        print(f"   -> {decision.recommended_action}")
        print(f"   reason: {decision.reason}")
        if decision.violations:
            print(f"   violations: {decision.violations}")
        audit.append(
            {
                "event": "decision",
                "signal": {
                    "source": sig.source,
                    "observed_at": sig.observed_at,
                    "confidence": sig.confidence,
                },
                "decision": decision.to_dict(),
                "competitor_risk": model.risk_for(sig.competitor),
            }
        )

    if args.sync_nodes and len(node_ids) > 1:
        for source, target in pairwise(node_ids):
            audit.append({"event": "node_sync", "source": source, "target": target})
            engine.sync_nodes(source, target)
    if args.quarantine_suspicious:
        for competitor in model.quarantine_candidates():
            model.quarantine(competitor)
            audit.append({"event": "competitor_quarantined", "competitor": competitor})

    recommendation = governor.recommendation()
    audit.append({"event": "constraint_review", **recommendation})
    print("-" * 72)
    print(f"Constraint review: {json.dumps(recommendation, sort_keys=True)}")
    print(f"Engine metrics: {json.dumps(engine.metrics(), sort_keys=True)}")
    print(f"Audit authenticated: {audit.authenticated}")
    print(f"Audit chain intact: {audit.verify_chain()}")
    print(f"Audit trail written to: {audit.path}")


if __name__ == "__main__":
    main()
