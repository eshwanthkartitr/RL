#!/usr/bin/env python3
"""
ReleaseOps side-by-side demo: naive / wasteful baselines vs. cautious heuristics
(story from judge brief — fast, no 1.7B model required).

Run locally:  python demo/gradio_app.py
Hugging Face:  set SDK to Gradio and entrypoint to this file, or: python app_gradio.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Avoid Gradio’s analytics POST (fails in some proxied / offline setups).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from demo.episode_runner import (
    policy_cautious_catch,
    policy_cautious_restraint,
    policy_naive_catch,
    policy_wasteful_restraint,
    run_episode,
)
from demo.narration import narrate_step

SCENARIOS = {
    "The Catch (disabled payment path)": {
        "family": "green_ci_disabled_payment_test",
        "seed": 5555,
        "difficulty": "medium",
        "archetype_mix": "shortcut_ci__careful_qa",
        "baseline": policy_naive_catch,
        "trained": policy_cautious_catch,
        "blurb": (
            "CI is green, but a **payment test is disabled in the diff**. "
            "The naive path approves the risky ship; the cautious path **inspects the PR** "
            "and **blocks** the bad proposal before a safe path can emerge."
        ),
    },
    "The Restraint (low-risk docs)": {
        "family": "careful_qa_safe",
        "seed": 7777,
        "difficulty": "low",
        "archetype_mix": "careful_qa__expert_rm",
        "baseline": policy_wasteful_restraint,
        "trained": policy_cautious_restraint,
        "blurb": (
            "A **low-risk documentation** triage and a **clean** ship request. "
            "The wasteful path **spends the review budget** on an unnecessary ticket inspection; "
            "the restrained path **approves the triage** and reserves budget for when it matters."
        ),
    },
}

DISCLAIMER = (
    "Heuristic policies on the same **ReleaseOps** simulator as your GRPO run — for a fast, "
    "readable story. For real before/after metrics, use your checkpoint and "
    "`compare_qwen17_eval.sh`."
)

TERMINAL_NICE = {
    "safe_ship": "safe_ship (release is rule-clean)",
    "unsafe_ship": "unsafe_ship (violation got through or bad ship outcome)",
    "missed_deadline": "missed_deadline (time ran out)",
}


def _terminal_pretty(t: str | None) -> str:
    if not t:
        return "incomplete / unknown"
    return TERMINAL_NICE.get(t, t or "—")


def format_column(
    title: str,
    tagline: str,
    log,
    action_history: list,
) -> str:
    b_spent = log.initial_budget - log.final_budget
    term = _terminal_pretty(log.terminal)
    parts: list[str] = [
        f"#### {title}",
        f"_{tagline}_",
        "",
        f"**Terminal** &nbsp;`{term}`  ·  **Steps** {log.steps}  ·  **Budget** {b_spent} / {log.initial_budget}",
        "",
        "###### Decision trace",
    ]
    for i, h in enumerate(action_history, 1):
        a = h["action"]
        parts.append(f"{i}. {narrate_step(a, h.get('result'))}")
    return "\n\n".join(parts)


def run_comparison(scenario_name: str) -> tuple[str, str, str]:
    key = (scenario_name or "").strip() or "The Catch (disabled payment path)"
    cfg = SCENARIOS.get(key)
    if not cfg:
        return "Error: unknown scenario", "Choose a valid scenario.", ""

    fam, seed = cfg["family"], cfg["seed"]
    d, arch = cfg["difficulty"], cfg["archetype_mix"]

    log_b, h_b = run_episode(
        fam, seed, cfg["baseline"], difficulty=d, archetype_mix=arch, max_steps=25
    )
    log_t, h_t = run_episode(
        fam, seed, cfg["trained"], difficulty=d, archetype_mix=arch, max_steps=25
    )

    intro = "\n\n".join(
        [
            f"### {key}",
            f"> {cfg['blurb']}",
            f"\n*Same scenario & seed (`{seed}`) — only the **supervision policy** changes.*\n",
        ]
    )
    left = format_column(
        "Baseline (failure / waste pattern)",
        "Eager to ship, or over-inspects low-risk work.",
        log_b,
        h_b,
    )
    right = format_column(
        "Supervisor (gated decisions)",
        "Inspects when risk warrants it; approves when it does not.",
        log_t,
        h_t,
    )
    return intro, left, right


def find_training_plot() -> str | None:
    for name in ("eval_chart.png", "grpo_plot.png", "metrics_plot.png"):
        p = ROOT / "outputs" / name
        if p.is_file():
            return str(p)
    return None


# --- UI: clean layout, slate/emerald, plenty of air ---
_APP_CSS = """
/* Shell */
.gradio-container, .gradio-container > .contain {
  max-width: 1080px !important;
  margin: 0 auto !important;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;
}
body.gradio_api .gradio-container {
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
  min-height: 100vh;
}
footer { display: none; }

/* Hero */
#ro-hero {
  text-align: center;
  padding: 2.25rem 1.5rem 1.5rem;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
  margin-bottom: 1.5rem;
}
#ro-hero h1 {
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #0f172a;
  margin: 0 0 0.75rem 0;
  line-height: 1.2;
}
#ro-hero p {
  margin: 0;
  color: #475569;
  font-size: 1.02rem;
  line-height: 1.55;
  max-width: 38rem;
  margin-left: auto;
  margin-right: auto;
}
#ro-hero .ro-kicker {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #047857;
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  margin-bottom: 0.75rem;
}

/* Controls */
#ro-controls {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.25rem 1.5rem 1.5rem;
  margin-bottom: 1.25rem;
}
#ro-controls .wrap { gap: 0.75rem; }
#ro-run-btn { margin-top: 0.5rem; }

/* Split columns */
#ro-panels { gap: 1.25rem; align-items: stretch; }
#ro-panels .gr-markdown, #ro-panels [data-testid="markdown"] {
  height: 100%;
}
#ro-baseline, #ro-supervisor {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.25rem 1.35rem 1.5rem;
  min-height: 22rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
#ro-baseline { border-top: 3px solid #f87171; }
#ro-supervisor { border-top: 3px solid #10b981; }
#ro-baseline h4, #ro-supervisor h4 {
  margin-top: 0;
  font-size: 0.95rem;
  color: #0f172a;
  font-weight: 600;
  letter-spacing: -0.01em;
}
#ro-baseline p, #ro-supervisor p { color: #64748b; }
#ro-baseline h6, #ro-supervisor h6 {
  margin: 1rem 0 0.5rem;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
}
#ro-baseline li, #ro-supervisor li {
  margin-bottom: 0.5rem;
  line-height: 1.5;
  color: #334155;
  font-size: 0.95rem;
}

#ro-intro {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.25rem 1.5rem 1.35rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
#ro-intro h3 { margin-top: 0; color: #0f172a; }
#ro-intro blockquote {
  border-left: 3px solid #10b981;
  margin: 0.5rem 0 0.75rem;
  padding: 0.5rem 0 0.5rem 1rem;
  color: #475569;
  background: #f8fafc;
  border-radius: 0 8px 8px 0;
}
#ro-intro p:last-child { margin-bottom: 0; }

/* Plot accordion */
#ro-plot-acc { border: 1px solid #e2e8f0 !important; border-radius: 12px; overflow: hidden; }

/* Foot */
.ro-foot {
  text-align: center;
  color: #64748b;
  font-size: 0.86rem;
  line-height: 1.5;
  padding: 1.25rem 0.5rem 0.5rem;
  border-top: 1px solid #e2e8f0;
  margin-top: 0.5rem;
}
.ro-foot code { background: #f1f5f9; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.8em; }
"""


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
        neutral_hue="slate",
        text_size="md",
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_400",
    )

    with gr.Blocks(
        title="ReleaseOps Arena",
        theme=theme,
        css=_APP_CSS,
    ) as app:
        gr.Markdown(
            """
<div id="ro-hero">
  <div class="ro-kicker">OpenEnv · GRPO · release supervision</div>
  <h1>When should your agents be trusted?</h1>
  <p>
    <strong>ReleaseOps</strong> trains a <em>supervisor</em> — not the worker. Same simulator as your runs:
    a live side-by-side story in plain language (no raw JSON). Use your checkpoint for real metrics.
  </p>
</div>
            """,
        )

        plot_p = find_training_plot()
        with gr.Accordion("Training metrics (optional image)", open=bool(plot_p), elem_id="ro-plot-acc"):
            if plot_p:
                gr.Image(
                    value=plot_p,
                    type="filepath",
                    show_label=False,
                    show_download_button=True,
                    height=320,
                )
            else:
                gr.Markdown(
                    "Add **`outputs/eval_chart.png`** in the project root to show your reward / tool curves here."
                )

        with gr.Group(elem_id="ro-controls"):
            scenario = gr.Radio(
                list(SCENARIOS.keys()),
                value="The Catch (disabled payment path)",
                label="Scenario",
                info="Two curated seeds — “Catch” and “Restraint”.",
            )
            run = gr.Button(
                "Run comparison",
                variant="primary",
                size="lg",
                elem_id="ro-run-btn",
            )

        out_intro = gr.Markdown(elem_id="ro-intro")
        with gr.Row(elem_id="ro-panels", equal_height=True):
            with gr.Column(elem_id="ro-baseline", scale=1):
                out_left = gr.Markdown()
            with gr.Column(elem_id="ro-supervisor", scale=1):
                out_right = gr.Markdown()

        with gr.Accordion("How the two stories differ", open=False):
            gr.Markdown(
                """
**The Catch** — green CI, but a **disabled payment** path in the diff. Baselines that approve blind risk **`unsafe_ship`**; a good supervisor **inspects the PR** and **blocks** under the right rule.

**The Restraint** — a **docs-only** triage plus a **clean** ship. Wasteful play burns **evidence / budget** on a harmless ticket; a restrained one **resolves triage** first and keeps budget for real risk.
                """
            )

        gr.Markdown(DISCLAIMER, elem_classes="ro-foot")
        if os.environ.get("GRADIO_VERBOSE", "").strip() in ("1", "true", "True"):
            gr.Markdown(
                f'<p class="ro-foot">Local eval: <code>compare_qwen17_eval.sh</code>  ·  <code>{ROOT}</code></p>',
            )

        run.click(
            fn=run_comparison,
            inputs=[scenario],
            outputs=[out_intro, out_left, out_right],
        )
        app.load(
            fn=run_comparison,
            inputs=[scenario],
            outputs=[out_intro, out_left, out_right],
        )
    return app


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    app = build_app()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
