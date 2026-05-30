# scripts/core/callbacks.py
import json
import os
from transformers import TrainerCallback

KL_WARN_THRESHOLD = 0.5
REWARD_COLLAPSE_STD_THRESHOLD = 0.01
REWARD_COLLAPSE_WINDOW = 50


class DiagnosticCallback(TrainerCallback):
    """Logs per-step training metrics to JSONL and warns on reward hacking / collapse."""

    def __init__(self, output_path: str = "results/diagnostics/training_curves.jsonl"):
        self.output_path = output_path
        self._file = None
        self._low_std_streak = 0

    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._file = open(self.output_path, "a")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, **logs}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        kl = logs.get("kl") or logs.get("kl_div")
        if kl is not None and kl > KL_WARN_THRESHOLD:
            print(
                f"\nWARNING: KL divergence = {kl:.3f} at step {state.global_step}. "
                f"Consider lowering learning_rate or stopping GRPO early."
            )

        reward_std = logs.get("rewards/std") or logs.get("reward_std")
        if reward_std is not None:
            if reward_std < REWARD_COLLAPSE_STD_THRESHOLD:
                self._low_std_streak += 1
                if self._low_std_streak >= REWARD_COLLAPSE_WINDOW:
                    print(
                        f"\nWARNING: Reward std < {REWARD_COLLAPSE_STD_THRESHOLD} "
                        f"for {REWARD_COLLAPSE_WINDOW} consecutive steps. Reward may have collapsed."
                    )
            else:
                self._low_std_streak = 0

    def on_train_end(self, args, state, control, **kwargs):
        if self._file:
            self._file.close()
            self._file = None
