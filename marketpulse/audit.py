from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditTrail:
    def __init__(
        self,
        path: str | Path,
        key: str | bytes | None = None,
        require_authentication: bool = False,
    ):
        self.path = Path(path)
        secret = key if key is not None else os.environ.get("MARKETPULSE_AUDIT_KEY")
        if require_authentication and not secret:
            raise ValueError(
                "MARKETPULSE_AUDIT_KEY is required for authenticated audit trails"
            )
        self._key = secret.encode() if isinstance(secret, str) else secret
        if self._key is not None and len(self._key) < 32:
            raise ValueError("MARKETPULSE_AUDIT_KEY must contain at least 32 bytes")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._last_hash = self._recover_last_hash()
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("audit trail is malformed") from exc
        if self.path.exists() and not self.verify_chain():
            raise ValueError("refusing to append to a tampered audit trail")

    @property
    def authenticated(self) -> bool:
        return self._key is not None

    @staticmethod
    def _canonical(entry: dict[str, Any]) -> bytes:
        return json.dumps(
            entry, sort_keys=True, default=str, separators=(",", ":")
        ).encode()

    @classmethod
    def _hash(cls, entry: dict[str, Any]) -> str:
        unsigned = {
            key: value for key, value in entry.items() if key not in {"hash", "mac"}
        }
        return hashlib.sha256(cls._canonical(unsigned)).hexdigest()

    @classmethod
    def _mac(cls, entry: dict[str, Any], key: bytes) -> str:
        unsigned = {key: value for key, value in entry.items() if key != "mac"}
        return hmac.new(key, cls._canonical(unsigned), hashlib.sha256).hexdigest()

    def _recover_last_hash(self) -> str:
        if not self.path.exists():
            return "GENESIS"
        last = "GENESIS"
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                last = entry.get("hash", last)
        return last

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(event, dict):
            raise TypeError("event must be a mapping")
        reserved = {"ts", "prev", "hash", "mac"}
        if reserved.intersection(event):
            raise ValueError("event contains reserved audit fields")
        entry: dict[str, Any] = {
            **event,
            "ts": datetime.now(timezone.utc).isoformat(),
            "prev": self._last_hash,
        }
        entry["hash"] = self._hash(entry)
        if self._key is not None:
            entry["mac"] = self._mac(entry, self._key)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry, default=str, separators=(",", ":")) + "\n")
        self._last_hash = entry["hash"]
        return entry

    def verify_chain(self) -> bool:
        if not self.path.exists():
            return True
        previous = "GENESIS"
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                claimed_hash = entry.get("hash")
                claimed_mac = entry.pop("mac", None)
            except (TypeError, ValueError, KeyError):
                return False
            if entry.get("prev") != previous:
                return False
            if self._hash(entry) != claimed_hash:
                return False
            if self._key is not None:
                if not isinstance(claimed_mac, str) or not hmac.compare_digest(
                    self._mac(entry, self._key), claimed_mac
                ):
                    return False
            elif claimed_mac is not None:
                return False
            previous = claimed_hash
        return True
