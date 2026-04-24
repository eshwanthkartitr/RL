from typing import List, Literal, Optional
from pydantic import BaseModel

class Proposal(BaseModel):
    id: str
    worker: str
    claim: str
    request: str
    refs: List[str]
    risk: List[str]
    confidence: Literal["low", "medium", "high", "very_high"]
    is_active: bool = True
    status: Literal["unresolved", "approved", "blocked"] = "unresolved"
    rule_id: Optional[str] = None # Filled if blocked
    
    # Hidden info
    true_violation_id: Optional[str] = None # If approving this violates a safety rule
    hidden_details: str = "" # Full evidence exposed when inspected

class WorkerStat(BaseModel):
    worker: str
    hint: str
    recent: dict # {"correct": int, "incorrect": int}
