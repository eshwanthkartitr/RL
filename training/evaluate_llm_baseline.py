import json
import argparse
import re
import sys
import time
from importlib.util import find_spec
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from releaseops_arena.tool_env import ReleaseOpsToolEnv

TORCH_MODEL_NAME = "Salesforce/xLAM-7b-fc-r"
MLX_MODEL_NAME = "models/mlx/xlam-7b-fc-r-4bit"
MAX_STEPS = 5
MAX_NEW_TOKENS = 128

TASK_INSTRUCTION = """
You are an expert release oversight supervisor composing exactly one function call.
Choose the next tool call for a software release review environment.
Inspect suspicious high-risk evidence before approving ship proposals.
Approve low-risk, well-supported proposals when inspection is not justified.
Block proposals that violate a release safety rule.
Use only IDs listed in the current observation or valid-id summary.
Never invent proposal, PR, CI run, ticket, or worker IDs.
Do not repeat an evidence-gathering action that already appears in action history.
If evidence_actions_remaining is 0, do not inspect or ask_worker; approve, block, or hold instead.
For payments-risk ship proposals, inspect the PR diff before approving because disabled tests appear in diffs.
If evidence shows skipped, disabled, commented, or removed payment tests, block the proposal with payments_tests_must_pass.
Do not explain your reasoning.
""".strip()

RAW_TASK_INSTRUCTION = """
You are a release oversight supervisor composing exactly one function call.
Choose the next tool call for a software release review environment.
Use only IDs that appear in the current observation.
Do not explain your reasoning.
""".strip()

FORMAT_INSTRUCTION = """
The output MUST strictly adhere to this JSON format, and NO other text MUST be included.
Return exactly one tool call in tool_calls.
```
{
  "tool_calls": [
    {"name": "func_name", "arguments": {"argument1": "value1"}}
  ]
}
```
""".strip()

AVAILABLE_TOOLS = [
    {
        "name": "inspect_pr_diff",
        "description": "Inspect a pull request diff for risky edits, disabled tests, or safety-rule violations.",
        "parameters": {
            "type": "object",
            "properties": {
                "pr_id": {
                    "type": "string",
                    "description": "Pull request id from the current observation refs.",
                }
            },
            "required": ["pr_id"],
        },
    },
    {
        "name": "inspect_ci_run",
        "description": "Inspect CI output for skipped tests, failing checks, or deploy gate status.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "CI run id from the current observation refs.",
                }
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "inspect_ticket",
        "description": "Inspect a ticket for issue severity, scope, and whether it is documentation-only or risky.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {
                    "type": "string",
                    "description": "Ticket id from the current observation refs.",
                }
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "ask_worker",
        "description": "Ask a worker for a compact explanation of evidence, policy, confidence, or risk.",
        "parameters": {
            "type": "object",
            "properties": {
                "worker_id": {
                    "type": "string",
                    "description": "Worker id from the current observation worker_stats.",
                },
                "question_type": {
                    "type": "string",
                    "enum": ["evidence_basis", "policy_basis", "confidence_basis", "risk_summary"],
                    "description": "Question to ask the worker.",
                },
            },
            "required": ["worker_id", "question_type"],
        },
    },
    {
        "name": "approve_proposal",
        "description": "Approve an unresolved proposal when its evidence is sufficient and safety rules are not violated.",
        "parameters": {
            "type": "object",
            "properties": {
                "proposal_id": {
                    "type": "string",
                    "description": "Proposal id from the current observation proposals.",
                }
            },
            "required": ["proposal_id"],
        },
    },
    {
        "name": "block_proposal",
        "description": "Block an unresolved proposal that violates or likely violates a release safety rule.",
        "parameters": {
            "type": "object",
            "properties": {
                "proposal_id": {
                    "type": "string",
                    "description": "Proposal id from the current observation proposals.",
                },
                "rule_id": {
                    "type": "string",
                    "enum": ["no_p1_open", "payments_tests_must_pass", "deploy_checks_before_ship"],
                    "description": "Safety rule that justifies blocking the proposal.",
                },
            },
            "required": ["proposal_id", "rule_id"],
        },
    },
    {
        "name": "hold_release",
        "description": "Delay the release when remaining unresolved risk is too high or required checks are unclear.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason_code": {
                    "type": "string",
                    "enum": ["awaiting_fix", "risk_too_high", "policy_unclear", "deploy_checks_pending"],
                    "description": "Reason for holding the release.",
                }
            },
            "required": ["reason_code"],
        },
    },
]

RULE_ID_ALIASES = {
    "S1": "no_p1_open",
    "S2": "payments_tests_must_pass",
    "S3": "deploy_checks_before_ship",
}


def extract_json_object(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def convert_to_xlam_tool(tools):
    if isinstance(tools, dict):
        return {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": {
                k: v for k, v in tools["parameters"].get("properties", {}).items()
            },
        }
    return [convert_to_xlam_tool(tool) for tool in tools]


def evidence_actions_from_history(action_history):
    used = {
        "pr_ids": set(),
        "ci_run_ids": set(),
        "ticket_ids": set(),
        "worker_ids": set(),
    }
    for item in action_history:
        action = item.get("action", {})
        if action.get("tool") == "inspect_pr_diff" and action.get("pr_id"):
            used["pr_ids"].add(action["pr_id"])
        elif action.get("tool") == "inspect_ci_run" and action.get("run_id"):
            used["ci_run_ids"].add(action["run_id"])
        elif action.get("tool") == "inspect_ticket" and action.get("ticket_id"):
            used["ticket_ids"].add(action["ticket_id"])
        elif action.get("tool") == "ask_worker" and action.get("worker_id"):
            used["worker_ids"].add(action["worker_id"])
    return used


def collect_valid_ids(state, action_history=None):
    ids = {
        "unresolved_proposal_ids": [],
        "pr_ids": [],
        "ci_run_ids": [],
        "ticket_ids": [],
        "worker_ids": [],
    }
    used = evidence_actions_from_history(action_history or [])
    for proposal in state["proposals"]:
        if proposal.status == "unresolved":
            ids["unresolved_proposal_ids"].append(proposal.id)
        for ref in proposal.refs:
            if ref.startswith("pr_"):
                ids["pr_ids"].append(ref)
            elif ref.startswith("ci_run_"):
                ids["ci_run_ids"].append(ref)
            elif ref.startswith("ticket_"):
                ids["ticket_ids"].append(ref)
    ids["worker_ids"] = [
        stat.worker if hasattr(stat, "worker") else stat["worker"]
        for stat in state["worker_stats"]
    ]
    ids = {key: sorted(set(value) - used.get(key, set())) for key, value in ids.items()}
    if state["evidence_actions_remaining"] <= 0:
        ids["pr_ids"] = []
        ids["ci_run_ids"] = []
        ids["ticket_ids"] = []
        ids["worker_ids"] = []
    return ids


def available_tools_for_state(state, valid_ids):
    if state["evidence_actions_remaining"] <= 0:
        allowed = {"approve_proposal", "block_proposal", "hold_release"}
    else:
        allowed = {tool["name"] for tool in AVAILABLE_TOOLS}
        if not valid_ids["pr_ids"]:
            allowed.discard("inspect_pr_diff")
        if not valid_ids["ci_run_ids"]:
            allowed.discard("inspect_ci_run")
        if not valid_ids["ticket_ids"]:
            allowed.discard("inspect_ticket")
        if not valid_ids["worker_ids"]:
            allowed.discard("ask_worker")
    return [tool for tool in AVAILABLE_TOOLS if tool["name"] in allowed]


def summarize_release_strategy(state):
    suggestions = []
    if state["evidence_actions_remaining"] <= 0:
        suggestions.append("No evidence actions remain; choose approve_proposal, block_proposal, or hold_release.")

    unresolved = [p for p in state["proposals"] if p.status == "unresolved"]
    for proposal in unresolved:
        refs = ", ".join(proposal.refs) if proposal.refs else "no refs"
        suggestions.append(
            f"{proposal.id}: request={proposal.request}; risk={proposal.risk}; refs={refs}; worker={proposal.worker}."
        )
        if proposal.request == "ship_release_candidate" and "payments" in proposal.risk:
            pr_refs = [ref for ref in proposal.refs if ref.startswith("pr_")]
            if pr_refs and state["evidence_actions_remaining"] > 0:
                suggestions.append(
                    f"Payments ship proposal {proposal.id} is suspicious; inspect_pr_diff on {pr_refs[0]} before approval."
                )
            else:
                suggestions.append(
                    f"Payments ship proposal {proposal.id} should be blocked with payments_tests_must_pass if evidence is missing or risky."
                )
    return suggestions


def summarize_evidence_history(state, action_history):
    if not action_history:
        return []

    evidence_text = "\n".join(str(item.get("result", "")) for item in action_history).lower()
    hints = []
    risky_payment_evidence = any(
        marker in evidence_text
        for marker in ["skip", "skipped", "disabled", "commented out", "removed payment", "pytest.mark.skip"]
    )
    if risky_payment_evidence:
        for proposal in state["proposals"]:
            if proposal.status == "unresolved" and proposal.request == "ship_release_candidate" and "payments" in proposal.risk:
                hints.append(
                    f"Evidence indicates skipped/disabled payment tests. Next call should block_proposal for {proposal.id} with rule_id payments_tests_must_pass."
                )
    return hints


def build_xlam_prompt(observation, tools, valid_ids, action_history, strategy_hints, tool_result=None):
    query_parts = [
        "Decide the next ReleaseOps tool call.",
        "Return only the JSON object requested by the format instruction.",
        "Use only these valid IDs:",
        json.dumps(valid_ids, indent=2),
        "Action history:",
        json.dumps(action_history[-6:], indent=2),
        "Decision hints:",
        json.dumps(strategy_hints, indent=2),
        f"Current observation:\n{observation}",
    ]
    if tool_result:
        query_parts.append(f"Previous tool result:\n{tool_result}")

    query = "\n\n".join(query_parts)
    tools = convert_to_xlam_tool(tools)
    return (
        f"[BEGIN OF TASK INSTRUCTION]\n{TASK_INSTRUCTION}\n[END OF TASK INSTRUCTION]\n\n"
        f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
        f"[BEGIN OF FORMAT INSTRUCTION]\n{FORMAT_INSTRUCTION}\n[END OF FORMAT INSTRUCTION]\n\n"
        f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    )


def build_raw_xlam_prompt(observation, tool_result=None):
    query_parts = [
        "Decide the next ReleaseOps tool call.",
        "Return only the JSON object requested by the format instruction.",
        f"Current observation:\n{observation}",
    ]
    if tool_result:
        query_parts.append(f"Previous tool result:\n{tool_result}")

    query = "\n\n".join(query_parts)
    tools = convert_to_xlam_tool(AVAILABLE_TOOLS)
    return (
        f"[BEGIN OF TASK INSTRUCTION]\n{RAW_TASK_INSTRUCTION}\n[END OF TASK INSTRUCTION]\n\n"
        f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
        f"[BEGIN OF FORMAT INSTRUCTION]\n{FORMAT_INSTRUCTION}\n[END OF FORMAT INSTRUCTION]\n\n"
        f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    )


def normalize_action(action):
    if not isinstance(action, dict):
        return None

    if "tool_calls" in action:
        tool_calls = action.get("tool_calls") or []
        if isinstance(tool_calls, list) and tool_calls:
            action = tool_calls[0]
        else:
            return None

    if "tool" not in action:
        if "name" in action:
            args = action.get("arguments", {})
            if isinstance(args, str):
                args = extract_json_object(args) or {}
            if isinstance(args, dict):
                action = {"tool": action["name"], **args}
        elif "tool_name" in action:
            args = action.get("arguments", {})
            if isinstance(args, str):
                args = extract_json_object(args) or {}
            if isinstance(args, dict):
                action = {"tool": action["tool_name"], **args}

    if "rule_id" in action:
        action["rule_id"] = RULE_ID_ALIASES.get(action["rule_id"], action["rule_id"])

    return action if "tool" in action else None

def parse_action(text):
    action = normalize_action(extract_json_object(text))
    if action:
        return action

    # Fallback heuristics
    if "approve_proposal" in text:
        match = re.search(r"p\d+", text)
        if match: return {"tool": "approve_proposal", "proposal_id": match.group(0)}
    elif "block_proposal" in text:
        proposal_match = re.search(r"p\d+", text)
        rule_match = re.search(
            r"S[123]|no_p1_open|payments_tests_must_pass|deploy_checks_before_ship",
            text,
        )
        if proposal_match:
            rule_id = RULE_ID_ALIASES.get(rule_match.group(0), rule_match.group(0)) if rule_match else "payments_tests_must_pass"
            return {"tool": "block_proposal", "proposal_id": proposal_match.group(0), "rule_id": rule_id}
    elif "inspect_pr_diff" in text:
        match = re.search(r"pr_\d+", text)
        if match: return {"tool": "inspect_pr_diff", "pr_id": match.group(0)}
    elif "inspect_ci_run" in text:
        match = re.search(r"ci_run_\d+", text)
        if match: return {"tool": "inspect_ci_run", "run_id": match.group(0)}
    elif "inspect_ticket" in text:
        match = re.search(r"ticket_\d+", text)
        if match: return {"tool": "inspect_ticket", "ticket_id": match.group(0)}
    elif "ask_worker" in text:
        worker_match = re.search(r"(ci_fixer|qa_triage|release_manager|sre|dev_\d+)", text)
        question_match = re.search(r"evidence_basis|policy_basis|confidence_basis|risk_summary", text)
        if worker_match:
            return {
                "tool": "ask_worker",
                "worker_id": worker_match.group(0),
                "question_type": question_match.group(0) if question_match else "evidence_basis",
            }
    elif "hold_release" in text:
        return {"tool": "hold_release", "reason_code": "risk_too_high"}
    return {"tool": "invalid", "text": text}


def get_terminal_reason(state):
    if isinstance(state, dict):
        return state.get("terminal_reason")
    return getattr(state, "terminal_reason", None)


def chat_prompt(tokenizer, messages, tokenize=False):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


class TorchGenerator:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        ).to(self.device)

    def generate(self, messages, max_new_tokens):
        text = chat_prompt(self.tokenizer, messages, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_tokens = inputs.input_ids.shape[1]
        return self.tokenizer.decode(
            outputs[0][prompt_tokens:],
            skip_special_tokens=True,
        )


class MlxGenerator:
    def __init__(self, model_name):
        from mlx_lm import generate, load

        self.generate_text = generate
        self.model, self.tokenizer = load(model_name)

    def generate(self, messages, max_new_tokens):
        prompt = chat_prompt(self.tokenizer, messages, tokenize=False)
        return self.generate_text(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            verbose=False,
        )


def select_backend(backend, mlx_model):
    if backend != "auto":
        return backend
    if find_spec("mlx_lm") and Path(mlx_model).exists():
        return "mlx"
    return "torch"


def load_generator(args):
    backend = select_backend(args.backend, args.mlx_model)
    model_name = args.mlx_model if backend == "mlx" else args.torch_model
    print(
        f"Loading {backend} model {model_name} for zero-shot evaluation...",
        flush=True,
    )

    start = time.perf_counter()
    generator = MlxGenerator(model_name) if backend == "mlx" else TorchGenerator(model_name)
    load_seconds = time.perf_counter() - start
    print(f"Loaded in {load_seconds:.1f}s", flush=True)
    return backend, generator, load_seconds


def run_zero_shot_baseline(args):
    backend, generator, load_seconds = load_generator(args)

    metrics = {
        "safe_ship": 0,
        "unsafe_ship": 0,
        "missed_deadline": 0,
        "invalid_actions": 0,
        "total_budget_spent": 0,
        "backend": backend,
        "eval_mode": args.eval_mode,
        "load_seconds": round(load_seconds, 2),
        "generation_seconds": 0.0,
        "false_blocks": 0,
        "true_blocks": 0,
    }
    
    data = [json.loads(line) for line in open("training/data/eval.jsonl")]
    test_data = data[:args.limit]
    
    for i, kwargs in enumerate(test_data):
        print(f"\n--- Episode {i+1} : {kwargs['family']} ---", flush=True)
        env = ReleaseOpsToolEnv()
        obs_str = env.reset(**kwargs)
        initial_budget = env.state["review_budget_remaining"]
        
        tool_result = None
        action_history = []
        
        for step in range(args.max_steps):
            if env.done:
                break

            obs_str = env._render_observation()
            if args.eval_mode == "guided_zero_shot":
                valid_ids = collect_valid_ids(env.state, action_history)
                available_tools = available_tools_for_state(env.state, valid_ids)
                strategy_hints = summarize_release_strategy(env.state)
                strategy_hints.extend(summarize_evidence_history(env.state, action_history))
                prompt = build_xlam_prompt(
                    obs_str,
                    available_tools,
                    valid_ids,
                    action_history,
                    strategy_hints,
                    tool_result,
                )
            else:
                prompt = build_raw_xlam_prompt(obs_str, tool_result)

            messages = [{"role": "user", "content": prompt}]
            gen_start = time.perf_counter()
            gen_text = generator.generate(messages, args.max_new_tokens)
            gen_seconds = time.perf_counter() - gen_start
            metrics["generation_seconds"] += gen_seconds
            action = parse_action(gen_text)
            
            print(f"Step {step+1} ({gen_seconds:.1f}s) Model Action: {action}", flush=True)
            
            try:
                if action.get("tool") == "approve_proposal":
                    resp = env.approve_proposal(action.get("proposal_id"))
                elif action.get("tool") == "block_proposal":
                    resp = env.block_proposal(
                        action.get("proposal_id"),
                        action.get("rule_id", "payments_tests_must_pass"),
                    )
                elif action.get("tool") == "inspect_pr_diff":
                    resp = env.inspect_pr_diff(action.get("pr_id"))
                elif action.get("tool") == "inspect_ci_run":
                    resp = env.inspect_ci_run(action.get("run_id"))
                elif action.get("tool") == "inspect_ticket":
                    resp = env.inspect_ticket(action.get("ticket_id"))
                elif action.get("tool") == "ask_worker":
                    resp = env.ask_worker(
                        action.get("worker_id"),
                        action.get("question_type", "evidence_basis"),
                    )
                elif action.get("tool") == "hold_release":
                    resp = env.hold_release(action.get("reason_code", "risk_too_high"))
                else:
                    resp = "Invalid tool call format."
                    metrics["invalid_actions"] += 1
            except Exception as e:
                resp = str(e)
                metrics["invalid_actions"] += 1
            
            tool_result = resp
            action_history.append({"action": action, "result": resp})
            
        # Log outcome
        reason = get_terminal_reason(env.state)
        if reason == "safe_ship": metrics["safe_ship"] += 1
        elif reason == "unsafe_ship": metrics["unsafe_ship"] += 1
        else: metrics["missed_deadline"] += 1
        metrics["total_budget_spent"] += initial_budget - env.state["review_budget_remaining"]
        metrics["false_blocks"] += env.metrics.get("false_blocks", 0)
        metrics["true_blocks"] += env.metrics.get("true_blocks", 0)

    metrics["generation_seconds"] = round(metrics["generation_seconds"], 2)
            
    print("\n=== ZERO-SHOT BASELINE MATRIX ===")
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "mlx"],
        default="auto",
        help="auto uses a local MLX model when present, otherwise torch.",
    )
    parser.add_argument("--torch-model", default=TORCH_MODEL_NAME)
    parser.add_argument("--mlx-model", default=MLX_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--eval-mode",
        choices=["guided_zero_shot", "raw_zero_shot"],
        default="guided_zero_shot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_zero_shot_baseline(parse_args())
