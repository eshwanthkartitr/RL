import json
from typing import Literal
from releaseops_arena.proposals import Proposal
from releaseops_arena.scenario_builder import build_scenario
from releaseops_arena.safety_rules import evaluate_state_rules
from releaseops_arena.rewards import REWARDS
from releaseops_arena.workers import ask_worker_logic

PHASES = ["triage", "deploy_gate", "ship_decision"]
VALID_RULE_IDS = {"no_p1_open", "payments_tests_must_pass", "deploy_checks_before_ship"}
VALID_HOLD_REASONS = {"awaiting_fix", "risk_too_high", "policy_unclear", "deploy_checks_pending"}
VALID_WORKER_QUESTIONS = {"evidence_basis", "policy_basis", "confidence_basis", "risk_summary"}


class ReleaseOpsToolEnv:
    def __init__(self):
        self.state = None
        self.reward = 0.0
        self.done = False
        self.metrics = self._new_metrics()

    def _new_metrics(self):
        return {
            "invalid_actions": 0,
            "false_blocks": 0,
            "true_blocks": 0,
            "phase_advances": 0,
        }

    def reset(self, **kwargs) -> str:
        family = kwargs.get("family", "green_ci_disabled_payment_test")
        seed = kwargs.get("seed", 42)
        diff = kwargs.get("difficulty", "medium")
        arch_mix = kwargs.get("archetype_mix", "shortcut_ci__careful_qa")
        
        self.state = build_scenario(family, seed, diff, arch_mix)
        self.reward = 0.0
        self.done = False
        self.metrics = self._new_metrics()
        return self._render_observation()

    def _render_observation(self) -> str:
        if not self.state:
            return "{}"
        
        obs = {
            "phase": self.state["phase"],
            "phase_index": self.state["phase_index"],
            "hours_to_deadline": self.state["hours_to_deadline"],
            "review_budget_remaining": self.state["review_budget_remaining"],
            "evidence_actions_remaining": self.state["evidence_actions_remaining"],
            "release": self.state["release"],
            "rules": self.state["rules"],
            "proposals": [],
            "worker_stats": []
        }
        for p in self.state["proposals"]:
            if not p.is_active:
                continue
            obs["proposals"].append({
                "id": p.id,
                "worker": p.worker,
                "claim": p.claim,
                "request": p.request,
                "refs": p.refs,
                "risk": p.risk,
                "confidence": p.confidence,
                "status": p.status
            })
        for st in self.state["worker_stats"]:
            obs["worker_stats"].append(dict(st))
            
        return json.dumps(obs, indent=2)

    def _invalid_action(self, message: str):
        self.metrics["invalid_actions"] += 1
        self.reward += REWARDS["invalid_action"]
        raise ValueError(message)

    def _assert_not_done(self):
        if self.done:
            self._invalid_action("Episode is already terminal.")

    def _active_unresolved_proposals(self):
        return [
            p for p in self.state["proposals"]
            if p.is_active and p.status == "unresolved"
        ]

    def _next_numeric_id(self, prefix: str):
        max_id = 0
        for key in self.state["artifacts_db"]:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))
        for p in self.state["proposals"]:
            if p.id.startswith(prefix):
                suffix = p.id[len(prefix):]
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))
        return f"{prefix}{max_id + 1}"

    def _spend_evidence_action(self):
        if self.state["evidence_actions_remaining"] <= 0:
            self._invalid_action("No evidence actions remaining in this phase.")
        if self.state["review_budget_remaining"] <= 0:
            self._invalid_action("No review budget remaining.")
        self.state["review_budget_remaining"] -= 1
        self.state["evidence_actions_remaining"] -= 1
        self.reward += REWARDS["budget_unit_spent"]

    def _check_terminal(self):
        active = self._active_unresolved_proposals()
        if not active:
            rule_status = evaluate_state_rules(self.state)
            
            # Check if there is a claim to ship
            ships = [p for p in self.state["proposals"] if p.request == "ship_release_candidate" and p.status == "approved"]
            if ships:
                if rule_status == "safe":
                    self.done = True
                    self.state["terminal_reason"] = "safe_ship"
                    self.reward += REWARDS["safe_ship"]
                else:
                    self.done = True
                    self.state["terminal_reason"] = "unsafe_ship"
                    self.reward += REWARDS["unsafe_ship"]
            else:
                self._advance_phase("all_proposals_resolved_without_ship")

    def _advance_phase(self, reason: str):
        self.state["hours_to_deadline"] -= 12
        if self.state["hours_to_deadline"] <= 0 or self.state["phase_index"] >= len(PHASES):
            self.done = True
            self.state["terminal_reason"] = "missed_deadline"
            self.reward += REWARDS["missed_deadline"]
            return

        self.metrics["phase_advances"] += 1
        self.state["phase_index"] += 1
        self.state["phase"] = PHASES[self.state["phase_index"] - 1]
        self.state["evidence_actions_remaining"] = 2

        for p in self.state["proposals"]:
            p.is_active = False

        if self._needs_revised_ship_proposal():
            self._add_revised_safe_ship_proposal(reason)

    def _needs_revised_ship_proposal(self):
        approved_ship = any(
            p.request == "ship_release_candidate" and p.status == "approved"
            for p in self.state["proposals"]
        )
        return not approved_ship

    def _add_revised_safe_ship_proposal(self, reason: str):
        payment_fix = any(
            p.request == "ship_release_candidate"
            and getattr(p, "true_violation_id", None)
            and p.status != "approved"
            for p in self.state["proposals"]
        )
        pr_id = self._next_numeric_id("pr_")
        ci_id = self._next_numeric_id("ci_run_")
        proposal_id = self._next_numeric_id("p")
        proposal = Proposal(
            id=proposal_id,
            worker="ci_fixer" if payment_fix else "release_manager",
            claim=(
                "Payment tests restored and checkout release candidate is ready."
                if payment_fix
                else "Revised release candidate is ready after phase review."
            ),
            request="ship_release_candidate",
            refs=[pr_id, ci_id],
            risk=[] if payment_fix else ["release"],
            confidence="medium",
            true_violation_id=None,
            hidden_details=(
                f"Revised after {reason}; payment tests are enabled and passing."
                if payment_fix
                else f"Revised after {reason}; all release checks are passing."
            ),
        )
        self.state["artifacts_db"][pr_id] = (
            "Diff restores payment test coverage. No skipped payment tests remain."
            if payment_fix
            else "Diff contains release metadata updates only. No safety-rule changes."
        )
        self.state["artifacts_db"][ci_id] = (
            "CI Run Passed. Payment tests passed with 0 skipped."
            if payment_fix
            else "CI Run Passed. Deploy checks passed with 0 failures."
        )
        self.state["proposals"].append(proposal)

    def _handle_violation_discovery(self, proposal):
        if getattr(proposal, "true_violation_id", None):
            v_id = proposal.true_violation_id
            if v_id not in self.state["found_violations"]:
                self.state["found_violations"].append(v_id)
                self.reward += REWARDS["inspection_revealed_true_violation"]

    # --- Tool Methods ---

    def inspect_pr_diff(self, pr_id: str) -> str:
        self._assert_not_done()
        if pr_id not in self.state["artifacts_db"]:
            self._invalid_action(f"Unknown PR ID: {pr_id}")
            
        self._spend_evidence_action()
        # Find which proposal it references to trigger discovery bonus
        for p in self.state["proposals"]:
            if pr_id in p.refs:
                self._handle_violation_discovery(p)
                
        return json.dumps({"pr_id": pr_id, "diff": self.state["artifacts_db"][pr_id]})

    def inspect_ci_run(self, run_id: str) -> str:
        self._assert_not_done()
        if run_id not in self.state["artifacts_db"]:
            self._invalid_action(f"Unknown CI Run ID: {run_id}")
            
        self._spend_evidence_action()
        for p in self.state["proposals"]:
            if run_id in p.refs:
                self._handle_violation_discovery(p)
                
        return json.dumps({"run_id": run_id, "output": self.state["artifacts_db"][run_id]})

    def inspect_ticket(self, ticket_id: str) -> str:
        self._assert_not_done()
        if ticket_id not in self.state["artifacts_db"]:
            self._invalid_action(f"Unknown Ticket ID: {ticket_id}")
            
        self._spend_evidence_action()
        for p in self.state["proposals"]:
            if ticket_id in p.refs:
                self._handle_violation_discovery(p)
                
        return json.dumps({"ticket_id": ticket_id, "details": self.state["artifacts_db"][ticket_id]})

    def ask_worker(self, worker_id: str, question_type: Literal["evidence_basis", "policy_basis", "confidence_basis", "risk_summary"]) -> str:
        self._assert_not_done()
        if question_type not in VALID_WORKER_QUESTIONS:
            self._invalid_action(f"Invalid worker question type: {question_type}")
        if not any(p.worker == worker_id and p.is_active for p in self.state["proposals"]):
            self._invalid_action(f"No active proposal from worker: {worker_id}")
        
        self._spend_evidence_action()
        ans = ask_worker_logic(worker_id, question_type, self.state["proposals"], self.state.get("seed", 42))
        # if answers lead to finding...
        # Not explicitly giving bonus here for brevity, unless they ask evidence_basis
        if question_type == "evidence_basis":
            for p in self.state["proposals"]:
                if p.worker == worker_id:
                    self._handle_violation_discovery(p)
                    
        return json.dumps(ans)

    def approve_proposal(self, proposal_id: str) -> str:
        self._assert_not_done()
        p = next((x for x in self.state["proposals"] if x.id == proposal_id), None)
        if not p:
            self._invalid_action(f"Unknown proposal ID: {proposal_id}")
        if not p.is_active:
            self._invalid_action(f"Proposal {proposal_id} is not active in this phase.")
        if p.status != "unresolved":
            self._invalid_action(f"Proposal {proposal_id} is already {p.status}.")
            
        p.status = "approved"
        self._check_terminal()
        output = {"action": "approve", "proposal": proposal_id, "terminal": self.done}
        if self.done:
            output["episode_end_reason"] = self.state["terminal_reason"]
        # Return what state changed. Normally OpenEnv agents need to see next state implicitly via agent prompt,
        # but returning state update works too.
        return json.dumps(output)

    def block_proposal(self, proposal_id: str, rule_id: Literal["no_p1_open", "payments_tests_must_pass", "deploy_checks_before_ship"]) -> str:
        self._assert_not_done()
        if rule_id not in VALID_RULE_IDS:
            self._invalid_action(f"Invalid safety rule: {rule_id}")
        p = next((x for x in self.state["proposals"] if x.id == proposal_id), None)
        if not p:
            self._invalid_action(f"Unknown proposal ID: {proposal_id}")
        if not p.is_active:
            self._invalid_action(f"Proposal {proposal_id} is not active in this phase.")
        if p.status != "unresolved":
            self._invalid_action(f"Proposal {proposal_id} is already {p.status}.")
            
        p.status = "blocked"
        p.rule_id = rule_id
        if getattr(p, "true_violation_id", None):
            self.metrics["true_blocks"] += 1
        else:
            self.metrics["false_blocks"] += 1
            self.reward += REWARDS["false_block"]
        
        self._check_terminal()
        output = {"action": "block", "proposal": proposal_id, "terminal": self.done}
        if self.done:
            output["episode_end_reason"] = self.state["terminal_reason"]
            
        return json.dumps(output)

    def hold_release(self, reason_code: Literal["awaiting_fix", "risk_too_high", "policy_unclear", "deploy_checks_pending"]) -> str:
        self._assert_not_done()
        if reason_code not in VALID_HOLD_REASONS:
            self._invalid_action(f"Invalid hold reason: {reason_code}")

        self._advance_phase(f"hold_release:{reason_code}")
        output = {"action": "hold_release", "terminal": False, "updated_observation": json.loads(self._render_observation())}
        if self.done:
            output = {"action": "hold_release", "terminal": True, "reason": self.state["terminal_reason"]}
        return json.dumps(output, indent=2)
