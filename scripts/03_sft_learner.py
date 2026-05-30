"""
Hours 3-4: SFT with Position-Balanced Data (LEARNER track)
CLI wrapper — training logic lives in scripts/core/train.py.
"""
import os
from scripts.core.data import build_training_dataset
from scripts.core.train import train_sft
from scripts.core.callbacks import DiagnosticCallback

train_dataset = build_training_dataset(balance_strategy="oversample")
print(f"Total training examples: {len(train_dataset)}")

diagnostic_cb = DiagnosticCallback("results/diagnostics/sft_training_curves.jsonl")

train_sft(
    model_path="CraneAILabs/ganda-gemma-1b",
    train_dataset=train_dataset,
    output_dir="./learner-full",
    lora_rank=16,
    num_train_epochs=3,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    callbacks=[diagnostic_cb],
)
