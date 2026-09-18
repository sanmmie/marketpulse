from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class CompetitiveSignal:
    competitor: str
    price_delta_pct: float
    promo_intensity: float
    sentiment_shift: float
    new_launch: bool = False
    confidence: float = 1.0
    source: str = "unknown"
    observed_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.competitor, str) or not self.competitor.strip():
            raise ValueError("competitor must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        self.price_delta_pct = self._finite("price_delta_pct", self.price_delta_pct)
        self.promo_intensity = self._finite("promo_intensity", self.promo_intensity)
        self.sentiment_shift = self._finite("sentiment_shift", self.sentiment_shift)
        self.confidence = self._finite("confidence", self.confidence)
        if not -1.0 <= self.price_delta_pct <= 1.0:
            raise ValueError("price_delta_pct must be between -1 and 1")
        if not 0.0 <= self.promo_intensity <= 1.0:
            raise ValueError("promo_intensity must be between 0 and 1")
        if not -1.0 <= self.sentiment_shift <= 1.0:
            raise ValueError("sentiment_shift must be between -1 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.new_launch, bool):
            raise TypeError("new_launch must be a boolean")
        if self.observed_at is not None and (
            not isinstance(self.observed_at, str) or not self.observed_at.strip()
        ):
            raise ValueError("observed_at must be a non-empty string when provided")

    @staticmethod
    def _finite(name: str, value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def to_input_data(self, baseline: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(baseline, dict):
            raise TypeError("baseline must be a mapping")
        price_war_signal = 1 if self.promo_intensity > 0.7 else 0
        return {
            "competitor_price_delta": self.price_delta_pct,
            "promo_intensity": self.promo_intensity,
            "sentiment_shift": self.sentiment_shift,
            "proposed_discount": max(0.0, -self.price_delta_pct),
            "projected_margin": float(baseline.get("projected_margin", 0.30))
            + self.price_delta_pct,
            "price_war_signal": price_war_signal,
            "price_war_confidence": self.confidence if price_war_signal else 0.0,
            "launch_flag": 1 if self.new_launch else 0,
            "signal_confidence": self.confidence,
            "signal_source": self.source,
            "observed_at": self.observed_at,
        }


def synthetic_stream(n: int = 8, seed: int = 7) -> list[CompetitiveSignal]:
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    competitors = ["AcmeCorp", "NorthwindRetail", "VertexGoods", "LumenCo"]
    return [
        CompetitiveSignal(
            competitor=rng.choice(competitors),
            price_delta_pct=round(rng.uniform(-0.22, 0.05), 3),
            promo_intensity=round(rng.uniform(0.0, 1.0), 3),
            sentiment_shift=round(rng.uniform(-1.0, 1.0), 3),
            new_launch=rng.random() < 0.25,
            confidence=round(rng.uniform(0.65, 1.0), 3),
            source="synthetic",
        )
        for _ in range(n)
    ]


def from_webhook(payloads: Iterable[dict[str, Any]]) -> list[CompetitiveSignal]:
    signals: list[CompetitiveSignal] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            raise TypeError("webhook payloads must be mappings")
        signals.append(
            CompetitiveSignal(
                competitor=payload["competitor"],
                price_delta_pct=payload["price_delta_pct"],
                promo_intensity=payload["promo_intensity"],
                sentiment_shift=payload["sentiment_shift"],
                new_launch=payload.get("new_launch", False),
                confidence=payload.get("confidence", 1.0),
                source=payload.get("source", "webhook"),
                observed_at=payload.get("observed_at"),
            )
        )
    return signals
