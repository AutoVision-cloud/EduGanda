#!/bin/bash
set -e

echo "=== EduGanda Setup ==="

# Work out of persistent volume if available, otherwise current dir
if [ -d "/workspace" ]; then
  cd /workspace
  echo "Using persistent volume at /workspace"
fi

# Clone or update repo
if [ -d "EduGanda" ]; then
  echo "Repo exists — pulling latest..."
  cd EduGanda && git pull
else
  echo "Cloning repo..."
  git clone https://github.com/AutoVision-cloud/EduGanda.git
  cd EduGanda
fi

# Install Python deps (use uv if available for faster installs, fallback to pip)
echo "Installing dependencies..."
if ! command -v uv &>/dev/null; then
  echo "Installing uv for faster dependency resolution..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

if command -v uv &>/dev/null; then
  uv pip install -r requirements.txt --system
else
  pip install -q -r requirements.txt
fi

# HuggingFace login (required for gated Gemma model)
echo ""
echo "=== HuggingFace Login ==="
echo "You need a HF token with access to google/gemma-3-1b-it."
echo "Get one at: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

# Optional: W&B login for experiment tracking
echo ""
read -p "Set up Weights & Biases experiment tracking? (y/n): " setup_wandb
if [ "$setup_wandb" = "y" ]; then
  echo "Get your API key at: https://wandb.ai/authorize"
  wandb login
  export WANDB_PROJECT="eduganda-extension"
  echo "export WANDB_PROJECT=eduganda-extension" >> ~/.bashrc
else
  export WANDB_MODE=disabled
  echo "export WANDB_MODE=disabled" >> ~/.bashrc
fi

# Create output directories
mkdir -p results/diagnostics results/ablations results/deploy

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run the pipeline in order:"
echo "  python scripts/01_explore_data.py"
echo "  python scripts/02_baseline_eval.py"
echo "  python scripts/03_sft_learner.py"
echo "  python scripts/04_grpo.py"
echo "  python scripts/05_merge.py"
echo "  python scripts/06_evaluate_all.py"
echo ""
echo "Then ablations:"
echo "  python scripts/ablations/run_ablation.py --axis lora_rank --value 8"
echo "  (see docs/superpowers/plans/ for the full execution order)"
