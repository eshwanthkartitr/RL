"""
Run short ReleaseOps episodes for UI demos: naive / wasteful baselines vs. cautious heuristics
that match the "two-seed" story (no LLM required — honest labels in the app).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from releaseops_arena.tool_env import ReleaseOpsToolEnv


@dataclass
class EpisodeLog:
    lines: list[str] = field(default_factory=list)
    steps: int = 0
    initial_budget: int = 0
    initial_evidence: int = 0
    terminal: Optional[str] = None
    final_budget: int = 0


def _first_ticket_ref(refs: list) -> Optional[str]:
    for ref in refs or []:
        if isinstance(ref, str) and ref.startswith("ticket_"):
            return ref
    return None


def _pr_ref(refs: list) -> Optional[str]:
    for ref in refs or []:
        if isinstance(ref, str) and ref.startswith("pr_"):
            return ref
    return None


def _inspected_prs(history: list) -> set[str]:
    s = set()
    for h in history:
        a = h.get("action") or {}
        if a.get("tool") == "inspect_pr_diff" and a.get("pr_id"):
            s.add(a["pr_id"])
    return s


def _inspected_tickets(history: list) -> set[str]:
    s = set()
    for h in history:
        a = h.get("action") or {}
        if a.get("tool") == "inspect_ticket" and a.get("ticket_id"):
            s.add(a["ticket_id"])
    return s


def _exec(env: ReleaseOpsToolEnv, action: dict) -> Any:
    t = action.get("tool")
    if t == "approve_proposal":
        return env.approve_proposal(action["proposal_id"])
    if t == "block_proposal":
        return env.block_proposal(action["proposal_id"], action.get("rule_id", "S2"))
    if t == "inspect_pr_diff":
        return env.inspect_pr_diff(action["pr_id"])
    if t == "inspect_ci_run":
        return env.inspect_ci_run(action["run_id"])
    if t == "inspect_ticket":
        return env.inspect_ticket(action["ticket_id"])
    if t == "ask_worker":
        return env.ask_worker(
            action["worker_id"],
            action.get("question_type", "evidence_basis"),
        )
    if t == "hold_release":
        return env.hold_release(action.get("reason_code", "risk_too_high"))
    raise ValueError(f"unknown tool {t}")


def policy_naive_catch(state: dict, history: list) -> dict:
    """Rushes to approve the payments ship line — the failure mode judges should see."""
    for p in state["proposals"]:
        if not p.is_active or p.status != "unresolved":
            continue
        if p.request == "ship_release_candidate" and "payments" in (p.risk or []):
            return {"tool": "approve_proposal", "proposal_id": p.id}
    for p in state["proposals"]:
        if p.is_active and p.status == "unresolved" and p.request == "ship_release_candidate":
            return {"tool": "approve_proposal", "proposal_id": p.id}
    return {"tool": "hold_release", "reason_code": "awaiting_fix"}


def policy_cautious_catch(state: dict, history: list) -> dict:
    """
    Encodes a competent supervisor: inspect the payments PR, then block the bad ship
    with the payment rule; then resolve other proposals.
    """
    hist = _inspected_prs(history)

    for p in state["proposals"]:
        if not p.is_active or p.status != "unresolved":
            continue
        if p.request == "ship_release_candidate" and "payments" in (p.risk or []):
            pr = _pr_ref(p.refs)
            if pr and pr not in hist and state["evidence_actions_remaining"] > 0:
                return {"tool": "inspect_pr_diff", "pr_id": pr}
            if pr in hist or state["evidence_actions_remaining"] <= 0:
                return {
                    "tool": "block_proposal",
                    "proposal_id": p.id,
                    "rule_id": "S2",
                }
    for p in state["proposals"]:
        if p.is_active and p.status == "unresolved" and p.request == "approve_triage":
            return {"tool": "approve_proposal", "proposal_id": p.id}
    for p in state["proposals"]:
        if p.is_active and p.status == "unresolved" and p.request == "ship_release_candidate":
            return {"tool": "approve_proposal", "proposal_id": p.id}
    return {"tool": "hold_release", "reason_code": "awaiting_fix"}


def policy_wasteful_restraint(state: dict, history: list) -> dict:
    """Over-inspects a harmless docs ticket first — burns budget (story: restraint scenario)."""
    t_ins = _inspected_tickets(history)
    for p in state["proposals"]:
        if not p.is_active or p.status != "unresolved":
            continue
        if p.request == "approve_triage" and "docs" in (p.risk or []):
            tid = _first_ticket_ref(p.refs)
            if (
                tid
                and tid not in t_ins
                and state["evidence_actions_remaining"] > 0
            ):
                return {"tool": "inspect_ticket", "ticket_id": tid}
    for p in state["proposals"]:
        if p.is_active and p.status == "unresolved" and p.request == "approve_triage":
            return {"tool": "approve_proposal", "proposal_id": p.id}
    for p in state["proposals"]:
        if p.is_active and p.status == "unresolved" and p.request == "ship_release_candidate":
            return {"tool": "approve_proposal", "proposal_id": p.id}
    return {"tool": "hold_release", "reason_code": "awaiting_fix"}


def policy_cautious_restraint(state: dict, history: list) -> dict:
    """Trusts the low-risk triage, approves it without wasteful evidence on docs-only work."""
    for p in state["proposals"]:
        if not p.is_active or p.status != "unresolved":
            continue
        if p.request == "approve_triage" and "docs" in (p.risk or []):
            return {"tool": "approve_proposal", "proposal_id": p.id}
    for p in state["proposals"]:
        if p.is_active and p.status == "unresolved" and p.request == "ship_release_candidate":
            return {"tool": "approve_proposal", "proposal_id": p.id}
    return {"tool": "hold_release", "reason_code": "awaiting_fix"}


PolicyFn = Callable[[Any, list], dict]


def run_episode(
    family: str,
    seed: int,
    policy: PolicyFn,
    difficulty: str = "medium",
    archetype_mix: str = "shortcut_ci__careful_qa",
    max_steps: int = 20,
) -> tuple[EpisodeLog, list[dict]]:
    env = ReleaseOpsToolEnv()
    env.reset(
        family=family,
        seed=seed,
        difficulty=difficulty,
        archetype_mix=archetype_mix,
    )
    log = EpisodeLog()
    log.initial_budget = int(env.state["review_budget_remaining"])
    log.initial_evidence = int(env.state["evidence_actions_remaining"])
    action_history: list[dict] = []
    for _ in range(max_steps):
        if env.done:
            break
        action = policy(env.state, action_history)
        try:
            result = _exec(env, action)
        except Exception as exc:  # noqa: BLE001
            result = f"Error: {exc}"
        action_history.append({"action": action, "result": result})
        log.steps += 1

    log.terminal = (env.state or {}).get("terminal_reason")
    if env.state is not None:
        log.final_budget = int(env.state["review_budget_remaining"])
    return log, action_history
