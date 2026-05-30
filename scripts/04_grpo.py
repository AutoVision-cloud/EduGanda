"""
Hours 5-6: GRPO (RL track)
CLI wrapper — training logic lives in scripts/core/train.py.
Skip this script if GRPO is unstable — LEARNER-only is a valid contribution.
"""
import os
from scripts.core.data import load_fln_dataset, build_grpo_prompts
from scripts.core.train import train_grpo
from scripts.core.callbacks import DiagnosticCallback

fln = load_fln_dataset("all")
grpo_dataset = build_grpo_prompts(fln, n=300)
print(f"GRPO prompts: {len(grpo_dataset)}")

diagnostic_cb = DiagnosticCallback("results/diagnostics/grpo_training_curves.jsonl")

train_grpo(
    model_path="./learner-full",
    grpo_dataset=grpo_dataset,
    output_dir="./grpo-full",
    max_steps=600,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    callbacks=[diagnostic_cb],
)
