from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Intent:
    goal: str
    constraints: dict[str, Any] = field(default_factory=dict)
    ethical_constraints: list[str] = field(default_factory=list)
    rationale: str = ""
    priority: str = "normal"
    version: str = "1"
    constraint_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    seasonal_adjustments: dict[str, dict[str, Any]] = field(default_factory=dict)
    recalibration: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("intent.goal must be a non-empty string")
        if not isinstance(self.constraints, dict):
            raise TypeError("intent.constraints must be a mapping")
        if not isinstance(self.ethical_constraints, list) or not all(
            isinstance(item, str) for item in self.ethical_constraints
        ):
            raise TypeError("intent.ethical_constraints must be a list of strings")
        if not isinstance(self.rationale, str):
            raise TypeError("intent.rationale must be a string")
        if not isinstance(self.priority, str) or not self.priority.strip():
            raise ValueError("intent.priority must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("intent.version must be a non-empty string")
        if not isinstance(self.constraint_profiles, dict):
            raise TypeError("intent.constraint_profiles must be a mapping")
        if not isinstance(self.seasonal_adjustments, dict):
            raise TypeError("intent.seasonal_adjustments must be a mapping")
        if not isinstance(self.recalibration, dict):
            raise TypeError("intent.recalibration must be a mapping")
        for name, value in self.constraints.items():
            if name in {"min_margin", "max_discount"} and (
                not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"intent.constraints.{name} must be between 0 and 1")
            if name == "no_price_war" and not isinstance(value, bool):
                raise ValueError("intent.constraints.no_price_war must be a boolean")
        for profile_name, profile in self.constraint_profiles.items():
            if not isinstance(profile_name, str) or not isinstance(profile, dict):
                raise TypeError("constraint profiles must map names to mappings")
        for season, adjustments in self.seasonal_adjustments.items():
            if not isinstance(season, str) or not isinstance(adjustments, dict):
                raise TypeError("seasonal adjustments must map names to mappings")

    def effective_constraints(
        self, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        context = context or {}
        constraints = dict(self.constraints)
        profile_name = context.get("constraint_profile") or context.get("segment")
        profile = self.constraint_profiles.get(profile_name, {})
        constraints.update(profile)
        season = context.get("season")
        adjustments = self.seasonal_adjustments.get(season, {})
        constraints.update(adjustments)
        return constraints

    def to_payload(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "constraints": self.effective_constraints(context),
            "ethical_constraints": list(self.ethical_constraints),
            "rationale": self.rationale,
            "priority": self.priority,
            "version": self.version,
            "constraint_profiles": {
                name: dict(profile)
                for name, profile in self.constraint_profiles.items()
            },
            "seasonal_adjustments": {
                name: dict(adjustments)
                for name, adjustments in self.seasonal_adjustments.items()
            },
            "recalibration": dict(self.recalibration),
            "updated_at": self.updated_at,
        }


@dataclass
class ConstraintGovernor:
    intent: Intent
    window_size: int = 20
    breach_rate_threshold: float = 0.25
    adjustment_step: float = 0.01
    min_margin_ceiling: float = 0.50
    max_discount_floor: float = 0.00
    outcomes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 0 <= self.breach_rate_threshold <= 1:
            raise ValueError("breach_rate_threshold must be between 0 and 1")
        if self.adjustment_step <= 0:
            raise ValueError("adjustment_step must be positive")

    def observe(self, score: float, hard_violations: list[str]) -> None:
        self.outcomes.append(
            {
                "score": max(0.0, min(1.0, float(score))),
                "hard_violations": list(hard_violations),
            }
        )
        if len(self.outcomes) > self.window_size:
            self.outcomes.pop(0)

    def recommendation(self) -> dict[str, Any]:
        if len(self.outcomes) < self.window_size:
            return {"status": "insufficient_data", "changes": {}}
        breaches = [
            outcome
            for outcome in self.outcomes
            if isinstance(outcome.get("hard_violations"), list)
            and outcome.get("hard_violations")
        ]
        rate = len(breaches) / len(self.outcomes)
        if rate < self.breach_rate_threshold:
            return {"status": "unchanged", "breach_rate": round(rate, 3), "changes": {}}
        changes: dict[str, float] = {}
        constraints = self.intent.constraints
        if any(
            "min_margin" in item
            for outcome in breaches
            for item in outcome["hard_violations"]
        ):
            changes["min_margin"] = min(
                self.min_margin_ceiling,
                float(constraints.get("min_margin", 0.0)) + self.adjustment_step,
            )
        if any(
            "max_discount" in item
            for outcome in breaches
            for item in outcome["hard_violations"]
        ):
            changes["max_discount"] = max(
                self.max_discount_floor,
                float(constraints.get("max_discount", 1.0)) - self.adjustment_step,
            )
        return {
            "status": "review_required",
            "breach_rate": round(rate, 3),
            "changes": changes,
        }


def load_intent(path: str | Path) -> Intent:
    data = yaml_safe_load(Path(path))
    if not isinstance(data, dict):
        raise TypeError("intent file must contain a mapping")
    constraints = data.get("constraints", {})
    profiles = data.get("constraint_profiles", {})
    seasonal = data.get("seasonal_adjustments", {})
    recalibration = data.get("recalibration", {})
    if not isinstance(constraints, dict):
        raise TypeError("constraints must be a mapping")
    if (
        not isinstance(profiles, dict)
        or not isinstance(seasonal, dict)
        or not isinstance(recalibration, dict)
    ):
        raise TypeError(
            "constraint profiles, seasonal adjustments, and recalibration must be mappings"
        )
    ethical_constraints = data.get("ethical_constraints", [])
    if not isinstance(ethical_constraints, list) or not all(
        isinstance(item, str) for item in ethical_constraints
    ):
        raise TypeError("ethical_constraints must be a list of strings")
    return Intent(
        goal=data.get("goal", ""),
        constraints=constraints,
        ethical_constraints=ethical_constraints,
        rationale=data.get("rationale", ""),
        priority=data.get("priority", "normal"),
        version=str(data.get("version", "1")),
        constraint_profiles=profiles,
        seasonal_adjustments=seasonal,
        recalibration=recalibration,
        updated_at=data.get("updated_at"),
    )


def yaml_safe_load(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text())


def publish_manifest(intent: Intent) -> dict[str, Any]:
    manifest = {
        "schema_version": "marketpulse.intent-manifest/v1",
        "declared_goal": intent.goal,
        "hard_constraints": intent.constraints,
        "constraint_profiles": intent.constraint_profiles,
        "seasonal_adjustments": intent.seasonal_adjustments,
        "ethical_constraints": intent.ethical_constraints,
        "rationale": intent.rationale,
        "priority": intent.priority,
        "version": intent.version,
        "updated_at": intent.updated_at,
        "autonomy_note": (
            "Engine may act autonomously only when intent_compliance_score "
            "meets or exceeds the configured floor and no hard violation is present. "
            "Otherwise it escalates."
        ),
    }
    canonical = json.dumps(manifest, sort_keys=True, default=str, separators=(",", ":"))
    manifest["manifest_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return manifest
