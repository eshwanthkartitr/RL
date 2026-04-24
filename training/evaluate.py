import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from releaseops_arena.tool_env import ReleaseOpsToolEnv
from releaseops_arena.baselines import naive_baseline, rule_baseline

def summarize_rollout(env, reward, initial_budget):
    reason = env.state.get("terminal_reason")
    return {
        "reward": reward,
        "safe_ship": 1 if reason == "safe_ship" else 0,
        "unsafe_ship": 1 if reason == "unsafe_ship" else 0,
        "missed_deadline": 1 if reason == "missed_deadline" else 0,
        "false_blocks": env.metrics.get("false_blocks", 0),
        "true_blocks": env.metrics.get("true_blocks", 0),
        "invalid_actions": env.metrics.get("invalid_actions", 0),
        "budget_spent": initial_budget - env.state["review_budget_remaining"],
    }


def aggregate(rows):
    if not rows:
        return {}
    return {
        "avg_reward": round(sum(row["reward"] for row in rows) / len(rows), 3),
        "safe_ship_rate": round(sum(row["safe_ship"] for row in rows) / len(rows), 3),
        "unsafe_ship_rate": round(sum(row["unsafe_ship"] for row in rows) / len(rows), 3),
        "missed_deadline_rate": round(sum(row["missed_deadline"] for row in rows) / len(rows), 3),
        "avg_false_blocks": round(sum(row["false_blocks"] for row in rows) / len(rows), 3),
        "avg_true_blocks": round(sum(row["true_blocks"] for row in rows) / len(rows), 3),
        "avg_invalid_actions": round(sum(row["invalid_actions"] for row in rows) / len(rows), 3),
        "avg_budget_spent": round(sum(row["budget_spent"] for row in rows) / len(rows), 3),
    }


def run_eval():
    print("Evaluating Baselines...")
    data = []
    with open("training/data/eval.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    naive_rows = []
    rule_rows = []
    
    for i, kwargs in enumerate(data):
        # Naive
        env_naive = ReleaseOpsToolEnv()
        env_naive.reset(**kwargs)
        naive_initial_budget = env_naive.state["review_budget_remaining"]
        r_naive = naive_baseline(env_naive)
        naive_rows.append(summarize_rollout(env_naive, r_naive, naive_initial_budget))
        
        # Rule
        env_rule = ReleaseOpsToolEnv()
        env_rule.reset(**kwargs)
        rule_initial_budget = env_rule.state["review_budget_remaining"]
        r_rule = rule_baseline(env_rule)
        rule_rows.append(summarize_rollout(env_rule, r_rule, rule_initial_budget))

    naive_summary = aggregate(naive_rows)
    rule_summary = aggregate(rule_rows)
    
    print(f"Evaluated {len(data)} seeds.")
    print("Naive Baseline:")
    print(json.dumps(naive_summary, indent=2))
    print("Rule Baseline:")
    print(json.dumps(rule_summary, indent=2))
    
    # Optional: Log to file or plotting directory
    results = {"naive": naive_summary, "rule": rule_summary}
    with open("outputs/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    run_eval()
