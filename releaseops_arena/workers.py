import random
from typing import Dict, Any

ANSWERS_EVIDENCE_SAFE = [
    "I checked the logs and the tests are thoroughly passing.",
    "The code diff only touches markdown files.",
    "I reviewed the PR and the UI tests are all green."
]

def ask_worker_logic(worker_id: str, question_type: str, state_proposals: list, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    # Find active proposal by worker_id in current phase
    worker_proposals = [p for p in state_proposals if p.worker == worker_id and p.is_active]
    if not worker_proposals:
        return {"tool": "ask_worker", "worker": worker_id, "error": f"No active proposal from {worker_id}"}
    
    p = worker_proposals[0]
    
    if question_type in ["evidence_basis", "risk_summary"]:
        if getattr(p, "true_violation_id", None):
            ans = p.hidden_details
        else:
            ans = rng.choice(ANSWERS_EVIDENCE_SAFE)
    else:
        ans = f"My confidence is {p.confidence} based on internal heuristics."
        
    return {
        "tool": "ask_worker",
        "worker": worker_id,
        "question_type": question_type,
        "answer": ans,
        "refs": p.refs,
        "confidence": p.confidence
    }
