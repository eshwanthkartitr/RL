#!/usr/bin/env python3
"""
Run a single ReleaseOps episode with a chat model: load a checkpoint, step the env, print answers.

- Local env (default): no Space needed; uses releaseops_arena.tool_env in-process.
- Remote env: `--env remote` uses your Space host (default: hiitsesh-new-gpu-space) or
  `RELEASEOPS_SPACE_URL` / `--space-url` — **host only** (e.g. `https://…hf.space`), not `…/outputs/ls`.

For manual "did RL help?" checks, run the same --family and --seed twice with
--torch-model (base) vs your Hub repo + --torch-subfolder best_by_loss, and
compare --jsonl-answers (or full --quiet) output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

import requests

# Default Space for this project (inference with remote /reset, /step). List files: {BASE}/outputs/ls
DEFAULT_HF_SPACE_BASE = "https://hiitsesh-new-gpu-space.hf.space"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# So `import evaluate_llm_baseline` resolves (same pattern as `python training/…py` scripts).
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

from releaseops_arena.tool_env import ReleaseOpsToolEnv

from evaluate_llm_baseline import (  # noqa: E402
    TorchGenerator,
    available_tools_for_state,
    build_raw_xlam_prompt,
    build_xlam_prompt,
    collect_valid_ids,
    get_terminal_reason,
    parse_action,
    sanitize_action,
    summarize_evidence_history,
    summarize_release_strategy,
)


def observation_dict_to_state(obs: dict) -> dict:
    """Rebuild a state-like object from HTTP/JSON observation (for guided prompt + sanitize)."""
    props: list[Any] = []
    for p in obs.get("proposals") or []:
        if isinstance(p, dict):
            d = dict(p)
            if "relevant_rule_ids" not in d and "possible_rule_violations" in d:
                d["relevant_rule_ids"] = d.get("possible_rule_violations") or []
            if "is_active" not in d:
                d["is_active"] = True
            props.append(types.SimpleNamespace(**d))
        else:
            props.append(p)
    return {
        "proposals": props,
        "evidence_actions_remaining": obs.get("evidence_actions_remaining", 0),
        "worker_stats": obs.get("worker_stats") or [],
    }


def _exec_local(env: ReleaseOpsToolEnv, action: dict) -> Any:
    t = action.get("tool")
    if t == "approve_proposal":
        return env.approve_proposal(action.get("proposal_id"))
    if t == "block_proposal":
        return env.block_proposal(
            action.get("proposal_id"),
            action.get("rule_id", "S2"),
        )
    if t == "inspect_pr_diff":
        return env.inspect_pr_diff(action.get("pr_id"))
    if t == "inspect_ci_run":
        return env.inspect_ci_run(action.get("run_id"))
    if t == "inspect_ticket":
        return env.inspect_ticket(action.get("ticket_id"))
    if t == "ask_worker":
        return env.ask_worker(
            action.get("worker_id"),
            action.get("question_type", "evidence_basis"),
        )
    if t == "hold_release":
        return env.hold_release(action.get("reason_code", "risk_too_high"))
    raise ValueError(f"unknown tool: {t}")


def flat_to_space_step(env_id: str, flat: dict) -> dict:
    tool = flat.get("tool")
    if not tool:
        raise ValueError("action missing tool")
    arguments = {k: v for k, v in flat.items() if k != "tool"}
    return {"env_id": env_id, "tool": tool, "arguments": arguments}


def run_episode(
    *,
    args: argparse.Namespace,
    generator: TorchGenerator,
) -> tuple[list[dict[str, Any]], Optional[str], float, float]:
    reset_kw = {
        "family": args.family,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "archetype_mix": args.archetype_mix,
    }
    if args.scenario_json:
        extra = json.loads(args.scenario_json)
        if not isinstance(extra, dict):
            raise SystemExit("--scenario-json must be a JSON object")
        reset_kw.update(extra)

    use_remote = args.env == "remote"
    if use_remote:
        base = args.space_url.rstrip("/")
        t0 = time.perf_counter()
        r = requests.post(f"{base}/reset", json=reset_kw, timeout=120)
        r.raise_for_status()
        payload = r.json()
        env_id = payload["env_id"]
        obs = payload["observation"]
        if not isinstance(obs, dict):
            obs = json.loads(obs) if isinstance(obs, str) else obs
        done = bool(payload.get("done"))
        t_reset = time.perf_counter() - t0
    else:
        env = ReleaseOpsToolEnv()
        t0 = time.perf_counter()
        env.reset(**reset_kw)
        t_reset = time.perf_counter() - t0
        done = bool(env.done)
        obs = json.loads(env.render_observation())
        env_id = None

    action_history: list[dict] = []
    tool_result: Any = None
    lines_out: list[dict[str, Any]] = []
    gen_time = 0.0
    if use_remote:
        last_terminal: Optional[str] = payload.get("terminal_reason")
    else:
        last_terminal = get_terminal_reason(env.state)  # type: ignore[union-attr]

    for step_i in range(args.max_steps):
        if done:
            break
        if use_remote:
            state = observation_dict_to_state(obs)
            obs_str = json.dumps(obs, indent=2)
        else:
            state = env.state
            obs_str = env._render_observation()

        if args.eval_mode == "guided_zero_shot":
            valid_ids = collect_valid_ids(state, action_history)
            tools = available_tools_for_state(state, valid_ids)
            hints = summarize_release_strategy(state)
            hints.extend(summarize_evidence_history(state, action_history))
            prompt = build_xlam_prompt(
                obs_str, tools, valid_ids, action_history, hints, tool_result
            )
        else:
            prompt = build_raw_xlam_prompt(obs_str, tool_result)

        messages = [{"role": "user", "content": prompt}]
        gen_start = time.perf_counter()
        model_text = generator.generate(messages, args.max_new_tokens)
        gen_dur = time.perf_counter() - gen_start
        gen_time += gen_dur
        parsed = parse_action(model_text)
        action, repaired = sanitize_action(parsed, state, action_history)

        if use_remote:
            body = flat_to_space_step(env_id, action)  # type: ignore[arg-type]
            sr = requests.post(
                f"{args.space_url.rstrip('/')}/step", json=body, timeout=120
            )
            sr.raise_for_status()
            out = sr.json()
            result = out.get("result")
            obs = out.get("observation") or {}
            if isinstance(obs, str):
                obs = json.loads(obs)
            done = bool(out.get("done"))
            last_terminal = out.get("terminal_reason")
        else:
            try:
                result = _exec_local(env, action)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                result = str(exc)
            last_terminal = get_terminal_reason(env.state)  # type: ignore[union-attr]
            done = bool(env.done)

        tool_result = result
        action_history.append({"action": action, "result": result})

        rec: dict[str, Any] = {
            "step": step_i,
            "seconds_generate": round(gen_dur, 3),
            "repaired": repaired,
            "raw_model_text": model_text,
            "parsed": parsed,
            "executed": action,
        }
        if args.show_observation:
            rec["observation_keys"] = list(obs.keys()) if isinstance(obs, dict) else None
        lines_out.append(rec)

    if use_remote:
        final_tr: Optional[str] = last_terminal
        if env_id and not done:
            try:
                requests.post(
                    f"{args.space_url.rstrip('/')}/close",
                    json={"env_id": env_id},
                    timeout=30,
                )
            except requests.RequestException:
                pass
    else:
        final_tr = get_terminal_reason(env.state)  # type: ignore[union-attr]

    return lines_out, final_tr, gen_time, t_reset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one ReleaseOps episode and print the model’s tool-calls (inference / manual check).",
    )
    p.add_argument(
        "--env",
        choices=["local", "remote"],
        default="local",
        help="local: in-process env. remote: env on a running Space (HTTP /reset, /step).",
    )
    p.add_argument(
        "--space-url",
        default="",
        help=(
            "Base URL of the env API (host only), e.g. https://hiitsesh-new-gpu-space.hf.space. "
            "Not /outputs/ls. If empty and --env remote, uses RELEASEOPS_SPACE_URL or the "
            "project default Space."
        ),
    )
    p.add_argument(
        "--torch-model",
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace id or local path; use your finetuned repo for RL checkpoint.",
    )
    p.add_argument(
        "--torch-subfolder",
        default="",
        help="Subfolder in a Hub repo, e.g. best_by_loss.",
    )
    p.add_argument(
        "--family", default="green_ci_disabled_payment_test"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--archetype-mix", default="shortcut_ci__careful_qa")
    p.add_argument(
        "--scenario-json",
        default="",
        help='Optional extra reset() kwargs as JSON object, e.g. \'{"family":"b", "seed":1}\'',
    )
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--eval-mode",
        choices=["guided_zero_shot", "raw_zero_shot"],
        default="guided_zero_shot",
    )
    p.add_argument(
        "--include-raw",
        action="store_true",
        help="In JSON/quiet output, include the raw model string (default off for --jsonl-answers).",
    )
    p.add_argument(
        "--jsonl-answers",
        action="store_true",
        help="One JSON object per line: executed action (+ optional raw with --include-raw), then a summary line.",
    )
    p.add_argument(
        "--only-executed",
        action="store_true",
        help="With --jsonl-answers, print only the post-sanitize tool call JSON per line (no step metadata).",
    )
    p.add_argument(
        "--show-observation",
        action="store_true",
        help="Add observation_keys to each JSON record (for debugging).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.env == "remote":
        base = (args.space_url or "").strip() or os.environ.get(
            "RELEASEOPS_SPACE_URL", ""
        ).strip()
        if not base:
            base = DEFAULT_HF_SPACE_BASE
        args.space_url = base.rstrip("/")

    sub = (args.torch_subfolder or "").strip() or None
    load_t0 = time.perf_counter()
    gen = TorchGenerator(args.torch_model, subfolder=sub)
    if not args.jsonl_answers and not args.only_executed:
        print(
            f"Loaded model in {time.perf_counter() - load_t0:.1f}s (device={gen.device!r})",
            file=sys.stderr,
        )

    lines, final_tr, gen_time, t_reset = run_episode(args=args, generator=gen)

    if args.jsonl_answers or args.only_executed:
        for rec in lines:
            if args.only_executed:
                out = rec["executed"]
            else:
                out = {
                    "step": rec["step"],
                    "executed": rec["executed"],
                }
                if args.include_raw:
                    out["raw_model_text"] = rec.get("raw_model_text", "")
            print(json.dumps(out, ensure_ascii=False))
        summary = {
            "terminal_reason": final_tr,
            "total_generation_seconds": round(gen_time, 3),
        }
        if args.env == "local":
            summary["reset_seconds"] = round(t_reset, 3)
        print(json.dumps(summary, ensure_ascii=False))
        return

    for rec in lines:
        rtxt = rec.get("raw_model_text", "")
        if len(rtxt) > 400 and not args.include_raw:
            rtxt = rtxt[:400] + "…"
        print(
            f"\n--- Step {rec['step'] + 1}  ({rec['seconds_generate']:.2f}s)  "
            f"repaired={rec['repaired']} ---\n{rtxt}\n  -> {rec['executed']}"
        )
    print(
        f"\nTerminal: {final_tr!r}  |  model generation: {gen_time:.2f}s total  |  reset: {t_reset:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
