import json
from releaseops_arena.tool_env import ReleaseOpsToolEnv

def naive_baseline(env: ReleaseOpsToolEnv) -> float:
    # Always approves active proposals
    while not env.done:
        obs = json.loads(env._render_observation())
        active_props = [p for p in obs["proposals"] if p["status"] == "unresolved"]
        if not active_props:
            # Nothing to do, ship or hold
            env.hold_release("awaiting_fix")
        else:
            for p in active_props:
                try:
                    env.approve_proposal(p["id"])
                except Exception:
                    pass
                if env.done:
                    break
    return env.reward

def rule_baseline(env: ReleaseOpsToolEnv) -> float:
    # Takes evidence if possible, blocks based on explicit danger, else approves
    while not env.done:
        obs = json.loads(env._render_observation())
        active_props = [p for p in obs["proposals"] if p["status"] == "unresolved"]
        if not active_props:
            env.hold_release("awaiting_fix")
        else:
            for p in active_props:
                # If refs available, inspect first
                if p["refs"] and env.state["evidence_actions_remaining"] > 0:
                    for ref in p["refs"]:
                        if ref.startswith("pr_"):
                            try: env.inspect_pr_diff(ref)
                            except Exception: pass
                            break
                        elif ref.startswith("ci_"):
                            try: env.inspect_ci_run(ref)
                            except Exception: pass
                            break
                # Rule logic: approve everything unless it has 'payments' risk
                if "payments" in p["risk"]:
                    try: env.block_proposal(p["id"], "payments_tests_must_pass")
                    except Exception: pass
                else:
                    try: env.approve_proposal(p["id"])
                    except Exception: pass
                    
                if env.done:
                    break
    return env.reward
