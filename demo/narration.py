"""
Turn ReleaseOps tool calls and env results into short, presentation-friendly lines.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

RULE_NICE = {
    "S1": "no P1 / critical bugs open (S1)",
    "S2": "payment tests must pass (S2)",
    "S3": "deploy checks before ship (S3)",
    "no_p1_open": "no P1 / critical bugs open (S1)",
    "payments_tests_must_pass": "payment tests must pass (S2)",
    "deploy_checks_before_ship": "deploy checks before ship (S3)",
}


def _shorten(s: str, n: int = 220) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_result(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def narrate_step(action: dict, result: Any) -> str:
    tool = action.get("tool", "?")
    r = _parse_result(result)
    rtext = str(r).lower() if r else ""

    if tool == "approve_proposal":
        pid = action.get("proposal_id", "?")
        if isinstance(r, dict) and r.get("episode_end_reason"):
            return f"**Approved** proposal `{pid}` — episode end: {r.get('episode_end_reason', '')}."
        return f"**Approved** proposal `{pid}`."

    if tool == "block_proposal":
        pid, rid = action.get("proposal_id"), action.get("rule_id", "S2")
        nice = RULE_NICE.get(str(rid), str(rid))
        return f"**Blocked** proposal `{pid}` under rule {nice}."

    if tool == "inspect_pr_diff":
        pr = action.get("pr_id", "")
        if any(
            w in rtext
            for w in ("disabled", "skip", "commented", "pytest.mark", "payment")
        ):
            return f"**Inspected** `{pr}` — diff shows **risky test behavior** (skipped/disabled or payment-path issues)."
        return f"**Inspected** `{pr}` for diff risk."

    if tool == "inspect_ci_run":
        return f"**Inspected** CI run `{action.get('run_id', '')}` for failures or skipped tests."

    if tool == "inspect_ticket":
        return f"**Inspected** ticket `{action.get('ticket_id', '')}` — spends review budget to read ticket text."

    if tool == "ask_worker":
        return (
            f"**Asked** worker `{action.get('worker_id', '')}` "
            f"({action.get('question_type', 'evidence')}) — budget spent on clarification."
        )

    if tool == "hold_release":
        return f"**Held** release: `{action.get('reason_code', 'risk_too_high')}`."

    return f"**{tool}** {action}."


def header_stats(budget0: int, budget: int, evidence0: int, evidence: int) -> str:
    spent = budget0 - budget
    return (
        f"*Review budget used:* **{spent}** / {budget0}  ·  "
        f"*Evidence actions left this phase:* **{evidence}** (started with {evidence0})"
    )
