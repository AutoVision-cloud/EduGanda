"""
Hour 7: Merge Ablation
Runs mergekit with 4 configurations: linear 80/20, 70/30, 60/40, and DARE-TIES 70/30.
Contribution #2: ablates merge ratios to characterise accuracy/fluency trade-off.

Requires ./learner-full and ./grpo-full (full merged checkpoints, not LoRA adapters).
If GRPO was skipped, comment out the grpo model entries and use learner-full for both.
"""

import os
import yaml
import subprocess

learner_path = os.path.abspath("./learner-full")
grpo_path = os.path.abspath("./grpo-full")

# Verify checkpoints exist
for path, name in [(learner_path, "learner-full"), (grpo_path, "grpo-full")]:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            f"If you skipped GRPO, duplicate learner-full as grpo-full or "
            f"remove grpo from the merge configs."
        )

# Linear merge configurations: (learner_weight, grpo_weight)
linear_configs = {
    "80-20": (0.8, 0.2),
    "70-30": (0.7, 0.3),
    "60-40": (0.6, 0.4),
}

for name, (lw, gw) in linear_configs.items():
    config = {
        "models": [
            {"model": learner_path, "parameters": {"weight": lw}},
            {"model": grpo_path, "parameters": {"weight": gw}},
        ],
        "merge_method": "linear",
        "dtype": "bfloat16",
    }
    config_path = f"configs/merge_{name}.yaml"
    output_path = f"./merged-{name}"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Running merge {name} → {output_path}")
    subprocess.run(["mergekit-yaml", config_path, output_path, "--cuda"], check=True)
    print(f"Done: {output_path}")

# DARE-TIES merge (what the paper calls "DARE/TIES of experts")
dare_config = {
    "models": [
        {"model": learner_path, "parameters": {"weight": 0.7, "density": 0.5}},
        {"model": grpo_path, "parameters": {"weight": 0.3, "density": 0.5}},
    ],
    "merge_method": "dare_ties",
    "base_model": learner_path,
    "dtype": "bfloat16",
}

with open("configs/merge_dare_ties.yaml", "w") as f:
    yaml.dump(dare_config, f)

print("Running DARE-TIES merge → ./merged-dare-ties")
subprocess.run(
    ["mergekit-yaml", "configs/merge_dare_ties.yaml", "./merged-dare-ties", "--cuda"],
    check=True,
)
print("Done: ./merged-dare-ties")
print("\nAll merges complete. Run 06_evaluate_all.py next.")
