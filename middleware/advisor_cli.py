"""One-shot stdin/stdout entry point for the STS2 reward advisor (SA6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.advisor_service import STS2AdvisorService  # noqa: E402
from pipeline.decks.advisor_schema import response_to_dict  # noqa: E402

_DEFAULT_SERVICE = STS2AdvisorService()


def evaluate_payload(payload: Any) -> Dict[str, Any]:
    """Evaluate a decoded request and return the versioned public result."""
    return _DEFAULT_SERVICE.evaluate_payload(payload)


def main() -> int:
    try:
        print(json.dumps(evaluate_payload(json.load(sys.stdin))))
        return 0
    except (ValueError, json.JSONDecodeError) as exc:
        print(json.dumps(response_to_dict({"status": "blocked", "diagnostics": str(exc)})))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
