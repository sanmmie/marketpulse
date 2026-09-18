from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from delta_net_async import (
        DeltaOS as _RealDeltaOS,  # type: ignore[import-not-found]
    )

    _DELTA_AVAILABLE = True
    _DELTA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    logger.warning("DeltaOS core not importable (%s). Using SIMULATION mode.", exc)
    _RealDeltaOS = None  # type: ignore[assignment]
    _DELTA_AVAILABLE = False
    _DELTA_IMPORT_ERROR = exc


def delta_available() -> bool:
    return _DELTA_AVAILABLE


def context_signature(env: dict[str, Any]) -> str:
    blob = json.dumps(env, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {value!r}")
    return result


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compute_delta(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, float]:
    if not isinstance(baseline, dict) or not isinstance(current, dict):
        raise TypeError("baseline and current must be mappings")
    keys = sorted(
        key
        for key in set(baseline) | set(current)
        if _is_finite_number(baseline.get(key, 0.0))
        and _is_finite_number(current.get(key, 0.0))
    )
    return {
        key: round(
            _finite_float(current.get(key, 0.0))
            - _finite_float(baseline.get(key, 0.0)),
            4,
        )
        for key in keys
    }


def score_compliance(
    proposal: dict[str, Any], intent: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(proposal, dict) or not isinstance(intent, dict):
        raise TypeError("proposal and intent must be mappings")
    constraints = intent.get("constraints", {}) or {}
    if not isinstance(constraints, dict):
        raise TypeError("intent.constraints must be a mapping")
    for name in ("min_margin", "max_discount"):
        if name in constraints:
            value = _finite_float(constraints[name], 0.0)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"intent.constraints.{name} must be between 0 and 1")
    if "no_price_war" in constraints and not isinstance(
        constraints["no_price_war"], bool
    ):
        raise TypeError("intent.constraints.no_price_war must be a boolean")

    score = 1.0
    violations: list[str] = []
    hard_violations: list[str] = []

    def breach(name: str, penalty: float, hard: bool = False) -> None:
        nonlocal score
        score -= penalty
        violations.append(name)
        if hard:
            hard_violations.append(name)

    proposed_discount = max(0.0, _finite_float(proposal.get("proposed_discount", 0.0)))
    max_discount = constraints.get("max_discount")
    if max_discount is not None:
        limit = _finite_float(max_discount)
        if proposed_discount > limit:
            breach(
                f"max_discount: {proposed_discount:.3f} > {limit:.3f}",
                0.35,
                hard=True,
            )

    projected_margin = proposal.get("projected_margin")
    min_margin = constraints.get("min_margin")
    if projected_margin is not None and min_margin is not None:
        margin = _finite_float(projected_margin)
        floor = _finite_float(min_margin)
        if margin < floor:
            breach(
                f"min_margin: {margin:.3f} < {floor:.3f}",
                0.35,
                hard=True,
            )

    price_war_signal = _finite_float(proposal.get("price_war_signal", 0.0))
    price_war_confidence = _finite_float(
        proposal.get("price_war_confidence", 1.0 if price_war_signal else 0.0),
        0.0,
    )
    price_war_threshold = _finite_float(
        proposal.get("price_war_confirmation_threshold", 0.80), 0.80
    )
    if not 0.0 <= price_war_threshold <= 1.0:
        raise ValueError("price_war_confirmation_threshold must be between 0 and 1")
    if constraints.get("no_price_war") and price_war_signal > 0:
        if price_war_confidence >= price_war_threshold:
            breach(
                "no_price_war: confirmed competitor price war detected", 0.45, hard=True
            )
        else:
            breach("no_price_war: unconfirmed competitor price war signal", 0.20)

    ethical_flags = intent.get("ethical_constraints", []) or []
    if not isinstance(ethical_flags, list):
        raise TypeError("intent.ethical_constraints must be a list")
    if "transparency" in ethical_flags and not str(intent.get("rationale", "")).strip():
        breach("transparency: missing declared rationale", 0.15, hard=True)

    return {
        "score": round(max(0.0, min(1.0, score)), 3),
        "violations": violations,
        "hard_violations": hard_violations,
        "allowed": not hard_violations,
    }


class _SimulatedDeltaOS:
    def __init__(self, context: dict[str, Any], intent: dict[str, Any]):
        self.context = dict(context)
        self.intent = dict(intent)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._metrics = {
            "cycles": 0,
            "holds": 0,
            "hard_violations": 0,
            "latency_total_ms": 0.0,
            "latency_max_ms": 0.0,
        }

    @property
    def engine_mode(self) -> str:
        return "simulation"

    def init_node(self, node_id: str, raw_env: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(raw_env, dict):
            raise TypeError("raw_env must be a mapping")
        node_context = dict(raw_env)
        node_context["node_id"] = node_id
        self._nodes[node_id] = node_context
        return {
            "node_id": node_id,
            "context_signature": context_signature(node_context),
            "state": "ready",
        }

    def run_cycle(
        self,
        input_data: dict[str, Any],
        raw_env: dict[str, Any],
        node_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if not isinstance(input_data, dict) or not isinstance(raw_env, dict):
            raise TypeError("input_data and raw_env must be mappings")
        baseline = raw_env.get("baseline", {}) or {}
        if not isinstance(baseline, dict):
            raise TypeError("raw_env.baseline must be a mapping")
        delta = compute_delta(baseline, input_data)
        compliance = score_compliance(input_data, self.intent)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        record = {
            "node_id": node_id,
            "state": "idle",
            "delta": delta,
            "context_signature": context_signature(raw_env),
            "intent_compliance_score": compliance["score"],
            "violations": compliance["violations"],
            "hard_violations": compliance["hard_violations"],
            "allowed": compliance["allowed"],
            "engine_mode": self.engine_mode,
            "latency_ms": latency_ms,
            "signal_confidence": _finite_float(
                input_data.get("signal_confidence", 1.0), 1.0
            ),
        }
        self._history.append(record)
        self._metrics["cycles"] += 1
        self._metrics["latency_total_ms"] += latency_ms
        self._metrics["latency_max_ms"] = max(
            self._metrics["latency_max_ms"], latency_ms
        )
        if not record["allowed"]:
            self._metrics["holds"] += 1
            self._metrics["hard_violations"] += len(record["hard_violations"])
        return record

    def sync_nodes(self, source: str, target: str) -> dict[str, Any]:
        if source not in self._nodes or target not in self._nodes:
            raise KeyError("both nodes must be initialized")
        shared = {
            **self._nodes[source],
            **self._nodes[target],
            "node_id": target,
        }
        self._nodes[target] = shared
        return {
            "source": source,
            "target": target,
            "context_signature": context_signature(shared),
        }

    def quarantine_node(self, node_id: str) -> dict[str, Any]:
        if node_id not in self._nodes:
            raise KeyError("unknown node")
        node = self._nodes.pop(node_id)
        node["quarantined"] = True
        return {"node_id": node_id, "status": "quarantined", "context": node}

    def metrics(self) -> dict[str, Any]:
        result = dict(self._metrics)
        result["latency_avg_ms"] = (
            round(result["latency_total_ms"] / result["cycles"], 3)
            if result["cycles"]
            else 0.0
        )
        return result


class _LiveDeltaOS:
    def __init__(self, context: dict[str, Any], intent: dict[str, Any]):
        if _RealDeltaOS is None:
            raise RuntimeError("DeltaOS core is unavailable")
        self.context = dict(context)
        self.intent = dict(intent)
        self._core = _RealDeltaOS(self.context, self.intent)
        self._fallback = _SimulatedDeltaOS(self.context, self.intent)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._metrics = {
            "cycles": 0,
            "holds": 0,
            "hard_violations": 0,
            "latency_total_ms": 0.0,
            "latency_max_ms": 0.0,
        }

    @property
    def engine_mode(self) -> str:
        return "live"

    def init_node(self, node_id: str, raw_env: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._core.init_node(node_id, raw_env)
        except Exception as exc:
            logger.warning(
                "Live node initialization failed; using simulation: %s",
                type(exc).__name__,
            )
            result = self._fallback.init_node(node_id, raw_env)
            self._nodes[node_id] = result
            return result
        finally:
            try:
                self._fallback.init_node(node_id, raw_env)
            except Exception as exc:
                logger.warning(
                    "Fallback node initialization failed: %s",
                    type(exc).__name__,
                )

    def run_cycle(
        self,
        input_data: dict[str, Any],
        raw_env: dict[str, Any],
        node_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            raw_result = self._core.run_cycle(input_data, raw_env, node_id)
            if isinstance(raw_result, tuple):
                core_record, feedback = raw_result
            else:
                core_record, feedback = raw_result, {}
            if not isinstance(core_record, dict):
                raise TypeError("DeltaOS run_cycle returned a non-mapping record")
            baseline = raw_env.get("baseline", {}) or {}
            delta = compute_delta(baseline, input_data)
            compliance = score_compliance(input_data, self.intent)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            self._metrics["cycles"] += 1
            self._metrics["latency_total_ms"] += latency_ms
            self._metrics["latency_max_ms"] = max(
                self._metrics["latency_max_ms"], latency_ms
            )
            if not compliance["allowed"]:
                self._metrics["holds"] += 1
                self._metrics["hard_violations"] += len(compliance["hard_violations"])
            return {
                "node_id": node_id,
                "state": core_record.get("state", "idle"),
                "delta": delta,
                "context_signature": core_record.get(
                    "context_signature", context_signature(raw_env)
                ),
                "intent_compliance_score": compliance["score"],
                "violations": compliance["violations"],
                "hard_violations": compliance["hard_violations"],
                "allowed": compliance["allowed"],
                "engine_mode": self.engine_mode,
                "latency_ms": latency_ms,
                "signal_confidence": _finite_float(
                    input_data.get("signal_confidence", 1.0), 1.0
                ),
                "core_intent_score": core_record.get("intent_score"),
                "core_feedback": feedback,
            }
        except Exception as exc:
            logger.warning(
                "Live cycle failed; using simulation: %s", type(exc).__name__
            )
            fallback = self._fallback.run_cycle(input_data, raw_env, node_id)
            fallback["engine_mode"] = "simulation_after_error"
            fallback["core_error"] = type(exc).__name__
            fallback["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self._metrics["cycles"] += 1
            self._metrics["latency_total_ms"] += fallback["latency_ms"]
            self._metrics["latency_max_ms"] = max(
                self._metrics["latency_max_ms"], fallback["latency_ms"]
            )
            if not fallback["allowed"]:
                self._metrics["holds"] += 1
                self._metrics["hard_violations"] += len(fallback["hard_violations"])
            return fallback

    def sync_nodes(self, source: str, target: str) -> dict[str, Any]:
        transmission = getattr(self._core, "transmission", None)
        if transmission is not None and hasattr(transmission, "sync_context"):
            return transmission.sync_context(source, target)
        return self._fallback.sync_nodes(source, target)

    def quarantine_node(self, node_id: str) -> dict[str, Any]:
        transmission = getattr(self._core, "transmission", None)
        if transmission is not None and hasattr(
            transmission, "quarantine_corrupted_nodes"
        ):
            return transmission.quarantine_corrupted_nodes(node_id)
        return self._fallback.quarantine_node(node_id)

    def metrics(self) -> dict[str, Any]:
        result = dict(self._metrics)
        result["latency_avg_ms"] = (
            round(result["latency_total_ms"] / result["cycles"], 3)
            if result["cycles"]
            else 0.0
        )
        return result


def build_engine(
    context: dict[str, Any], intent: dict[str, Any]
) -> _LiveDeltaOS | _SimulatedDeltaOS:
    if _DELTA_AVAILABLE and _RealDeltaOS is not None:
        logger.info("Using LIVE DeltaOS core.")
        return _LiveDeltaOS(context, intent)
    logger.info("Using SIMULATED DeltaOS core.")
    return _SimulatedDeltaOS(context, intent)
