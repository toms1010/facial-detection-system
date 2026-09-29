"""Privacy helpers for telemetry.

The hardware panel may show a hostname, and that is a machine identifier. The
privacy default is to keep it out of logs and exports, so snapshots can be
shared in a bug report without leaking the host name.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visionai.hardware.hardware_bridge import HardwareSnapshot

REDACTED = "redacted"

_SENSITIVE_KEYS = ("hostname", "kernel", "architecture")


def redact_snapshot(snapshot: HardwareSnapshot) -> HardwareSnapshot:
    """Replace machine-identifying fields in place, per the privacy settings."""
    if getattr(snapshot, "hostname", ""):
        snapshot.hostname = REDACTED
    return snapshot


def redact_mapping(payload: dict) -> dict:
    """Recursively redact sensitive keys in a nested mapping."""
    if not isinstance(payload, dict):
        return payload
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _SENSITIVE_KEYS and isinstance(value, str) and value:
            result[key] = REDACTED
        elif isinstance(value, dict):
            result[key] = redact_mapping(value)
        elif isinstance(value, list):
            result[key] = [
                redact_mapping(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def scrub_text(text: str, scrub_ips: bool = True) -> str:
    """Remove IP addresses and absolute home paths from free-form text."""
    scrubbed = re.sub(r"/home/[^/\s]+", f"/home/{REDACTED}", text)
    if scrub_ips:
        scrubbed = _IPV4.sub(REDACTED, scrubbed)
    return scrubbed
