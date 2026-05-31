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

# Create a persistent venv on the network volume (survives pod restarts)
VENV_DIR="/workspace/venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "export VIRTUAL_ENV=$VENV_DIR" >> ~/.bashrc
echo "export PATH=$VENV_DIR/bin:\$PATH" >> ~/.bashrc
echo "source $VENV_DIR/bin/activate" >> ~/.bashrc

# Install uv for fast package management
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Installer for packages — uv if available, pip fallback
_install() {
  if command -v uv &>/dev/null; then
    uv pip install "$@"
  else
    pip install -q "$@"
  fi
}

# Install PyTorch for CUDA 12.4 (compatible with RunPod driver 12040)
echo "Installing PyTorch (cu124)..."
_install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install remaining deps
echo "Installing other dependencies..."
_install transformers==4.50.3 trl==0.12.0 peft==0.14.0 \
    accelerate datasets bitsandbytes scikit-learn scipy \
    matplotlib wandb huggingface_hub mergekit pyyaml llama-cpp-python

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

# Add project root to PYTHONPATH so `from scripts.core.x import y` works
PROJ_DIR="$(pwd)"
export PYTHONPATH="$PROJ_DIR"
echo "export PYTHONPATH=\"$PROJ_DIR\"" >> ~/.bashrc

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
