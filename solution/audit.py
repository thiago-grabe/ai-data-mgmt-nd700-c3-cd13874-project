"""Per-query audit trail.

Every query records which agent answered it, which ethical check ran, what that
check found, and whether the response was modified as a result. Without this the
safeguards leave no evidence that they ran on any particular request — you can
show the code exists, but not that it fired.

One JSON object per line so the log is appendable, greppable, and readable by
pandas without a parser.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(os.environ.get("AUDIT_LOG_PATH", "logs/audit_log.jsonl"))


class AuditLog:
    def __init__(self, path: Path = DEFAULT_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        query: str,
        route: str,
        router_method: str,
        checks=None,
        response_modified: bool = False,
        error: str | None = None,
    ) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "route": route,
            "router_method": router_method,
            "checks": checks or [],
            "response_modified": response_modified,
            "error": error,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def summarize_bias(findings) -> dict:
    return {
        "check": "fairlearn_bias_audit",
        "outcome": "disparity_flagged" if findings else "within_thresholds",
        "finding_count": len(findings or []),
        "details": [
            f"{f['table']}.{f['column']} by {f['demographic_column']}: "
            f"{f['metric']}={f['value']:.4f}"
            for f in (findings or [])
        ],
    }


def summarize_pii(findings) -> dict:
    return {
        "check": "presidio_pii_detection",
        "outcome": "pii_detected_and_redacted" if findings else "no_pii_detected",
        "finding_count": len(findings or []),
        # Entity types and confidences only. Writing the detected PII values into
        # the audit log would recreate the exposure the check exists to prevent.
        "details": [
            f"{f['entity_type']} (confidence={f['score']:.2f})"
            for f in (findings or [])
        ],
    }


def summarize_content_safety(findings, blocked=None) -> dict:
    flagged = [f for f in (findings or []) if f.get("flagged")]
    return {
        "check": "azure_content_safety",
        "outcome": "unsafe_content_excluded" if flagged else "all_images_safe",
        "images_analyzed": len(findings or []),
        "images_blocked": len(blocked or []),
        "details": [
            f"{f['image']}: " + ", ".join(f"{k}={v}" for k, v in f["severities"].items())
            for f in (findings or [])
        ],
    }


def summarize_anomaly(findings, refused: bool = False) -> dict:
    return {
        "check": "timeseries_reliability_guardrail",
        "outcome": (
            "answer_withheld_unreliable_data" if refused
            else "anomalies_flagged" if findings
            else "no_anomalies"
        ),
        "finding_count": len(findings or []),
        "details": [
            f"{f['neighborhood']} {f['metric']} at {f['timestamp']}: "
            f"{f['value']:.2f} (z={f['zscore']:.2f})"
            for f in (findings or [])
        ],
    }
