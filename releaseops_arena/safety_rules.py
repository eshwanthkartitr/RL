def evaluate_state_rules(state) -> str:
    # Just a simple rule evaluator based on state.proposals that were approved.
    for p in state["proposals"]:
        if p.status == "approved" and getattr(p, "true_violation_id", None):
            return "violation"
    return "safe"
