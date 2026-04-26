# ReleaseOps Arena — full technical & narrative context (for humans *and* agents)

*This file is the **single canonical brief** for the project: what it is, how it is implemented, how training/eval/REST/Space work, and where the numbers come from. If any metric disagrees with a fresh `outputs/*.json` on disk, **trust the file**.*

**Agent / assistant bundle (JSON):** [outputs/agent_project_knowledge.json](outputs/agent_project_knowledge.json) — same as below plus **`rubric_and_evidence_index`** (requirement → file/URL), **`grpo_pilot_run`**, **`finetuned_qwen3_inference_eval`**, and baselines. Use it for structured review or a quick checklist pass.

**Primary code paths in this monorepo**

- **Boiler_plate (root)**: [openenv.yaml](openenv.yaml), [releaseops_arena/](releaseops_arena/) (env + baselines), [training/train_grpo.py](training/train_grpo.py) (older defaults may differ; see below).
- **Hugging Face Space / GPU line**: [New_gpu_space/training/train_grpo.py](New_gpu_space/training/train_grpo.py), [New_gpu_space/releaseops_arena/server.py](New_gpu_space/releaseops_arena/server.py) (REST + training child process), [New_gpu_space/SPACE_RUNBOOK.md](New_gpu_space/SPACE_RUNBOOK.md).

---

## 1. One-liner and thesis

**Most agent demos train an AI to do the work. ReleaseOps Arena trains an AI to decide when other AIs should be trusted.**

The loop you must remember:

**conflicting specialist proposals + a limited oversight budget + hard release safety rules**

Deeper spec: [ref.md](ref.md) (MVP design, mechanics, future work). OpenEnv “contract” (which Python classes back client/action/observation): [openenv.yaml](openenv.yaml).

| Concept | In code |
|--------|---------|
| **Client (HTTP to Space)** | `ReleaseOpsEnvClient` in `releaseops_arena.client` |
| **Action** | `ReleaseOpsAction` — `tool` + `arguments` (Pydantic) |
| **Observation** | `ReleaseOpsObservation` / rendered JSON in tool-env |
| **Stateful sim** | `ReleaseOpsToolEnv` in [releaseops_arena/tool_env.py](releaseops_arena/tool_env.py) |

The **trainable** agent is a **control-plane supervisor**, not “another release engineer agent” that writes code: it **allocates inspection budget** and **gates ship** under rules.

---

## 2. World model: phases, state, and observation JSON

### Phases and progression

`ReleaseOpsToolEnv` uses a fixed **phase list**: `triage` → `deploy_gate` → `ship_decision` (see `PHASES` in [tool_env.py](releaseops_arena/tool_env.py)). Episodes are **procedural** from `build_scenario(family, seed, difficulty, archetype_mix)`; each `reset()` samples world content consistent with a **narrative family** (e.g. CI green because a **payment test was disabled**).

### What the model *sees* (observation)

Observations are **structured JSON** (not free prose), e.g.:

- `phase`, `phase_index`, `hours_to_deadline`
- `review_budget_remaining`, `evidence_actions_remaining` (scarcity: you cannot inspect forever)
- `release` / `release_checks` (facts the env may reveal)
- `rules` (safety context)
- `proposals[]`: worker id, claim, request, refs, risk, confidence, `possible_rule_violations`, etc.
- `worker_stats[]`

Rendering is in `render_observation()`; default `reset()` kwargs include `family`, `seed`, `difficulty`, `archetype_mix`.

### Action format

Tool calls must match the **OpenEnv** surface exposed to TRL via `ReleaseOpsGRPOEnv` (training) or the HTTP client. Tools include: `inspect_pr_diff`, `inspect_ci_run`, `inspect_ticket`, `ask_worker` (with `question_type` in a fixed enum), `approve_proposal`, `block_proposal` (with a **safety `rule_id`**), `hold_release` (with a **reason_code**). Invalid IDs or impossible actions feed **penalties** in metrics and rewards.

### Safety rules (IDs you must not confuse)

Canonical rule IDs in the env include: `S1`↔`no_p1_open`, `S2`↔`payments_tests_must_pass`, `S3`↔`deploy_checks_before_ship` (see [safety_rules.py](releaseops_arena/safety_rules.py)). The engine evaluates `release_facts` (e.g. `open_p1_bug`, `payment_tests_disabled`, `deploy_checks_passed`) into **violations**; proposals can be **blocked** with reference to these rules. **Block** when justified; **false blocks** (blocking without a true underlying violation) are tracked in `metrics["false_blocks"]`.

### Terminal reasons (how eval labels outcomes)

The env reaches **terminal** states with a `terminal_reason` such as `safe_ship`, `unsafe_ship`, or `missed_deadline` — see [training/evaluate.py](training/evaluate.py) `summarize_rollout()` which maps that to per-episode flags for `safe_ship`, `unsafe_ship`, `missed_deadline`, plus `false_blocks`, `true_blocks`, `invalid_actions`, `budget_spent`.

---

## 3. Reward structure (shaping, not a single “chat score”)

Source of truth: [releaseops_arena/rewards.py](releaseops_arena/rewards.py) `REWARDS` table:

| Event | Weight (example) | Meaning for learning |
|-------|------------------|--------------------|
| `safe_ship` | +1.0 | Shipped with rules satisfied. |
| `unsafe_ship` | −1.0 | Shipped into violation / bad state. |
| `missed_deadline` | −0.6 | Process did not close in time. |
| `budget_unit_spent` | −0.05 per unit | **Inspect/ask** has a cost; encourages selective oversight. |
| `invalid_action` | −0.25 | Malformed or disallowed tool use. |
| `false_block` | −0.15 | Blocked when rule did not actually apply. |
| `inspection_revealed_true_violation` | +0.2 | Incentivize **useful** evidence gathering. |

GRPO **reads the scalar `env.reward`** at the end of each environment turn via `reward_func` (list comprehension over env instances) in `train_grpo.py` — the reward is **not** human preference labels; it is **this** simulator’s economics.

**Compatibility mode (only for broken/old TRL):** if `GRPOTrainer` has **no** `environment_factory`, training can fall back to `compatibility_reward_func`, which heuristically scores **text** for tool-name substrings. That path is **not** a valid substitute for true env RL; the trainer will **refuse** to start unless you pass `--allow-compatibility-reward` (see error message in [training/train_grpo.py](training/train_grpo.py) `main()`). **For judges: require TRL with OpenEnv GRPO + `environment_factory`.**

---

## 4. Dataset generation and train/eval splits

[training/make_dataset.py](training/make_dataset.py) builds **JSONL** rows. Each line is a training example with at least:

- `prompt` — chat turn list (system + user) for the supervisor.
- `family` — which procedural scenario class.
- `seed`, `difficulty`, `archetype_mix` — passed into `env.reset(**kwargs)`.

**Family catalog (`FAMILY_CONFIG`)** encodes e.g.:

- `green_ci_disabled_payment_test` — CI is “green” in a way that is **too good to be true** (disabled payment tests).
- `qa_undercalls_p1_checkout_bug` — QA understates a serious bug class.
- `release_manager_ship_before_evidence` — pressure to ship before deploy evidence.
- `careful_qa_safe` — lower difficulty “cleaner” path.

**Splits (important for *generalization* claims):**

- `TRAIN_FAMILIES` — e.g. `green_ci_disabled_payment_test`, `qa_undercalls_p1_checkout_bug`, `careful_qa_safe`.
- `UNSEEN_EVAL_FAMILIES` — e.g. `release_manager_ship_before_evidence` — held out for eval JSON that measures **unseen** family performance.

`merge_jsonl` can combine `eval_seen` + `eval_unseen` for aggregate reporting.

`train_grpo.py` calls `ensure_dataset()`; if the train file is missing, it **runs** `make_dataset.py` to generate it.

---

## 5. GRPO + TRL wiring (the actual training algorithm)

- **Library:** Hugging Face **TRL** `GRPOTrainer` with **Group Relative Policy Optimization** — policy gradient style updates using grouped samples; see TRL’s OpenEnv documentation for the version you pin in [requirements.txt](requirements.txt) / [New_gpu_space/requirements.txt](New_gpu_space/requirements.txt).
- **Environment factory:** `ReleaseOpsGRPOEnv` wrapps `ReleaseOpsToolEnv` and exposes **only** the tool methods TRL should bind as **named Python tools** (docstrings become tool descriptions). See class `ReleaseOpsGRPOEnv` in [training/train_grpo.py](training/train_grpo.py).
- **Reward:** `reward_funcs=[reward_func]` where `reward_func` returns each wrapped env’s `.reward`.
- **GRPOConfig** keys used include: `max_steps`, `num_generations`, `gradient_accumulation_steps`, `max_prompt_length`, `max_completion_length`, `learning_rate`, `logging_steps`, and when supported `env_kwargs_keys=["family","seed","difficulty","archetype_mix"]` so each batch row can **reset** a distinct scenario.
- **Dataset:** `datasets.load_dataset("json", data_files=train_file)`; each row’s metadata flows into the env reset.

**New_gpu_space (production Space) training script** extends the above with, among other things: **bf16**, **8-bit paged AdamW**, **gradient checkpointing** for 1.7B on L4, `chat_template_kwargs` for Qwen3, `BestLossCheckpointCallback`, optional **`--hub-model-repo` / `hub_upload`** for post-train push — see [New_gpu_space/training/train_grpo.py](New_gpu_space/training/train_grpo.py).

**Model defaults (do not mix them up):**

- **New_gpu_space** defaults `--model-name` to **`Qwen/Qwen3-0.6B`** for cost-effective Space runs; **1.7B** is the serious tier for `compare` / pilot.
- **Root** [training/train_grpo.py](training/train_grpo.py) in this tree may still default **`Qwen/Qwen2.5-0.5B-Instruct`** — treat **New_gpu_space** as the line you ship to HF; align versions when merging to `eshwanthkartitr/RL`.

### Qwen3-specific gotcha

Qwen3 chat templates may inject a **`<think>`**-style “thinking” block unless disabled. The Space training path builds a tokenizer with **`enable_thinking: false`** in chat template kwargs so the model emits **parseable tool calls**, not a hidden scratchpad that breaks the parser.

### Why 1.7B is the “serious” tier (engineering, not bragging)

- On **L4 22GB**, 1.7B + GRPO + generations needs **bf16** + **paged 8-bit optimizer** + **gradient checkpointing**; otherwise OOM.
- The **compare** story: [compare_qwen17_eval.sh](compare_qwen17_eval.sh) runs `evaluate_llm_baseline.py` for **Qwen3-1.7B zero-shot** vs **finetuned** checkpoint (`best_by_loss` locally or `hiitsesh/releaseops-grpo-1.7b-best` on Hub with subfolder `best_by_loss`).

**Published training narrative (any qualified base size):** ~**100** GRPO steps, reward **dip then rise** to a stable positive value (example **~3.09** in README), **tool-failure rate ~0** at convergence — [images/Training.png](images/Training.png).

---

## 6. Baselines and evaluation modes

### Hand-coded baselines (no LLM)

[training/evaluate.py](training/evaluate.py) runs `naive_baseline`, `rule_baseline`, and `phase_aware_rule_baseline` from [releaseops_arena/baselines.py](releaseops_arena/baselines.py) over the same `reset(**kwargs)` rows as the dataset, then aggregates in [outputs/eval_results.json](outputs/eval_results.json).

Example **overall** (from a checked-in run): naive **`safe_ship_rate` ~0.04**, **`unsafe_ship_rate` ~0.96**; rules do better on some slices but can still break on **unseen** families — the point is to show **headroom** for **learned** policies.

### LLM + tool-calling baselines and GRPO model eval

[training/evaluate_llm_baseline.py](training/evaluate_llm_baseline.py) drives `ReleaseOpsToolEnv` with a **torch** or **MLX** backend, composes a **strict JSON** `tool_calls` output format, and tracks the same high-level stats (`safe_ship`, `unsafe_ship`, `false_blocks`, `repaired_actions`, `total_budget_spent`, etc.). It includes a strong **xLAM-7B**-style function-calling baseline name in **defaults** for “big agentic model” comparison — the README’s narrative contrasts a **smaller** GRPO-tuned Qwen on **this** env vs large instruction-tuned **general** agents on **general** tasks.

**Space eval API** ([New_gpu_space/releaseops_arena/eval_api.py](New_gpu_space/releaseops_arena/eval_api.py)):

- `POST /api/run-eval?limit=&model_id=&subfolder=` — async subprocess to the same `evaluate_llm_baseline.py` (default Hub model id for the 1.7B **finetuned** artifact + `best_by_loss` subfolder).
- `GET /api/get-eval-results` — serves `outputs/eval_api_result.json` when complete.

---

## 7. REST API for training and artifacts (Space architecture)

**Design:** the FastAPI app does **not** reimplement SGD. It **`subprocess.Popen`’s** `python training/train_grpo.py ...` with the **same** arguments you would use locally, pipes stdout+stderr to a **line-buffered** log, and sets **container-safe** environment variables. **One training at a time** (lock; second start returns “already running”).

### `GET /` — machine-readable index

Returns JSON listing `/health`, `/docs`, static **pitch_video**, **notebook** URLs, `ui.flow`, and **every** `/train/...` and `/outputs/...` route so judges and agents can **discover** everything without reading source.

### Training routes (typical use)

| Route | Role |
|-------|------|
| `GET/POST /train/smoke` | Tiny GRPO run for CI/Sanity. |
| `GET/POST /train/pilot` | Query parameters: `max_steps`, `num_generations`, `gradient_accumulation_steps`, `max_completion_length`, `learning_rate`, `logging_steps`, `model_name` (**URL-encode** `/` as `%2F` for e.g. `Qwen/Qwen3-1.7B`), `bf16`, `best_loss_dir`, optional `hub_model_repo`, `hub_upload_include`. |
| `GET /train/status` | Subprocess state + elapsed. |
| `GET/POST /train/kill` | Stop child. |
| `GET /train/logs` | Tail text log. |
| `GET /train/metrics`, `/train/summary`, `/train/live` | Log-history JSON; summary stats; time series for plots. |
| `GET /train/plot.png`, `/train/plot` | PNG and auto-refresh HTML. |
| `GET/POST /train/push_to_hub?repo_id=...&path=...` | Manual Hub upload of `outputs/...` checkpoint trees. |

**Live metrics path:** `LiveLogHistoryCallback` writes `trainer.state.log_history` to JSON on each `on_log` / `on_save` / `on_train_end` so HTTP always sees **current** loss/reward/tokens without waiting for training to finish.

### Child-process environment (Space hardening)

Before spawn, the server sets (see [server.py](New_gpu_space/releaseops_arena/server.py) `_start_training`):

- `PYTHONUNBUFFERED=1` — **streaming** logs to `/train/logs`.
- `TORCH_COMPILE_DISABLE=1` — avoids dynamo/inductor duplicate registration issues on some images.
- `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` under writable `outputs/`.
- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME` — avoid **`/.cache`** and **`PermissionError`** when `HOME` is `/` or empty.
- If `HOME` is bad, set a synthetic home under the HF root — fixes `getpass` / `getpwuid` failures for torch/inductor.

### Artifact endpoints (runtime, not git “Files”)

- `GET /outputs/ls` — list what exists **on disk in the running container** (Hugging Face “Files” tab is **git**, not this).
- `GET /outputs/file?path=...` — download a single file.
- `GET /outputs/archive?path=...` — tarball a folder (e.g. grab `best_by_loss` before the Space is wiped).

`SPACE_RUNBOOK.md` documents `HF_TOKEN` secrets and example `curl` for pilot + `push_to_hub`.

### Environment HTTP for rollouts (OpenEnv)

`POST /reset`, `POST /step` (and related) are the **judge** path for driving the same env as training without local Python — [releaseops_arena/client.py](releaseops_arena/client.py) wraps the wire format. Use the `*.hf.space` **app** URL, not the `huggingface.co/spaces/...` **page** URL (README warns about this).

---

## 8. Hugging Face / submission constraints (agent checklist)

- **openenv-core** in requirements; env implementation is **your** `tool_env.py` — you **compose** on OpenEnv, you don’t fork the framework.
- **Training evidence:** provide plots / `outputs/*.json` from real runs; **no** giant binary videos in the repo.
- **Space** must be **public and runnable** at the exact submitted URL: [https://huggingface.co/spaces/hiitsesh/New_gpu_space](https://huggingface.co/spaces/hiitsesh/New_gpu_space).
- **Checkpoints:** store on **Hugging Face Hub** (or S3), **not** multi‑GB in Git; README uses **raw GitHub** for static images when the Hub repo can’t take PNGs.
- **Colab** for judges: [notebooks/ReleaseOps_final_walkthrough.ipynb](https://colab.research.google.com/github/eshwanthkartitr/RL/blob/main/notebooks/ReleaseOps_final_walkthrough.ipynb) (mirrored on Space as blob).

---

## 9. Reference links (stable URLs)

| Resource | URL |
|----------|-----|
| Live Space | [hiitsesh/New_gpu_space](https://huggingface.co/spaces/hiitsesh/New_gpu_space) |
| GitHub (RL) | [eshwanthkartitr/RL](https://github.com/eshwanthkartitr/RL) |
| Short pitch (YouTube) | [YouTube Shorts](https://www.youtube.com/shorts/OxfBH7jDOwg) |
| Episode flow (HTML) | [releaseops_episode_flow.html](https://github.com/eshwanthkartitr/RL/blob/main/demo/releaseops_episode_flow.html) |
| Judge / organizer doc | [Google Doc from README](https://docs.google.com/document/d/1Odznuzwtb1ecDOm2t6ToZd4MuMXXfO6vWUGcxbC6mFs/edit?tab=t.0#bookmark=kix.2dz0x0nie3me) |
| TRL OpenEnv | [Hugging Face TRL OpenEnv docs](https://huggingface.co/docs/trl/openenv) |

---

## 10. Glossary (for quick agent parsing)

| Term | Meaning here |
|------|----------------|
| **OpenEnv** | A pattern/spec for *tool-using* RL envs with a discoverable `openenv.yaml` and pip `openenv-core`. |
| **GRPO** | Group Relative Policy Optimization (TRL) — the RL algorithm in use. |
| **Supervisor** | The only trainable policy; workers are simulated. |
| **Family** | A procedural *class* of release scenarios. |
| **Archetype mix** | Which worker personas dominate (e.g. shortcut vs careful). |
| **evidence_actions_remaining** | Hard cap on how many **inspect/ask** tools can be used before the model must **approve / block / hold**. |
| **S1 / S2 / S3** | Short names for the three key safety rules (map to `no_p1_open`, `payments_tests_must_pass`, `deploy_checks_before_ship`). |
| **best_by_loss** | Checkpoint dir updated whenever training loss **improves** — not necessarily “best reward” unless correlated. |
| **Compatibility reward** | Text-heuristic **fallback** when TRL lacks `environment_factory` — **not** a submission-grade signal. |

---

## 11. Suggested “next work” (honest)

- Tighter **unseen** eval + **confidence intervals** on rates.
- **Pin** and periodically **upgrade** `trl` + `openenv-core` and re-run smoke + 1.7B compare.
- Optional **Rubric coverage:** multi-agent, oversight, long-horizon, tool use — this env hits all of them; document **which JSON fields** in observations justify each in your writeup.

---

## 12. Deliverable index (organizer “NOTE 1” crosswalk)

This mirrors the [README](README.md) checklist in one place. Any reviewer (including automated tooling) can follow **requirement → artifact** without digging through the tree.

| # | What organizers typically ask | Where to verify in this project |
|---|--------------------------------|----------------------------------|
| 1 | OpenEnv-based env, not a one-off ad-hoc sim | [openenv.yaml](openenv.yaml), [releaseops_arena/tool_env.py](releaseops_arena/tool_env.py), `openenv-core` in [requirements.txt](requirements.txt) |
| 2 | Real RL path + TRL; reproducible | [training/train_grpo.py](training/train_grpo.py), [notebooks/ReleaseOps_final_walkthrough.ipynb](notebooks/ReleaseOps_final_walkthrough.ipynb) + Colab link in README |
| 3 | Evidence of a training run (curves, metrics) | [images/Training.png](images/Training.png), [outputs/agent_project_knowledge.json](outputs/agent_project_knowledge.json) `grpo_pilot_run` |
| 4 | Public writeup and/or under-2 min video, no huge binaries in the Hub | [blog.md](blog.md) + [README](README.md) + YouTube in judge table |
| 5 | Live Hugging Face Space | [https://huggingface.co/spaces/hiitsesh/New_gpu_space](https://huggingface.co/spaces/hiitsesh/New_gpu_space) and `https://<space>.hf.space` for the API |
| 6 | README-level story (problem, env, results, links) | [README](README.md), [New_gpu_space/README.md](https://huggingface.co/spaces/hiitsesh/New_gpu_space/blob/main/README.md) on Hub |

**Themes to state once in a summary sentence:** multi-agent conflict (proposals in obs), **oversight** (supervisor is what you train), long-horizon + **tool** use (phases, budget, `ReleaseOpsGRPOEnv` tools).

For machine-friendly extraction of the same map plus metrics, use [outputs/agent_project_knowledge.json](outputs/agent_project_knowledge.json) key `rubric_and_evidence_index`.

---

*When editing this file, keep claims **traceable** to: `ref.md`, `rewards.py`, `tool_env.py`, `make_dataset.py`, `train_grpo.py` (root vs `New_gpu_space/`), `evaluate.py` / `evaluate_llm_baseline.py`, `server.py`, and `outputs/*.json`.*

**Update path:** after `compare_qwen17_eval.sh`, you can add `outputs/eval_zeroshot_qwen1.7b.json` / `outputs/eval_finetuned_rl.json` to mirror the same stats already summarized in [outputs/agent_project_knowledge.json](outputs/agent_project_knowledge.json).
