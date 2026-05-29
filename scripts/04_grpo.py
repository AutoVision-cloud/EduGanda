"""
Hours 5-6: GRPO (RL track)
Fine-tunes the LEARNER model with reinforcement learning using the Crane AI Labs reward model.
Saves a full checkpoint at ./grpo-full for mergekit.

NOTE: If GRPO diverges or the reward model behaves unexpectedly, skip this script.
A LEARNER-only model with fixed position bias is a valid contribution on its own.
"""

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOTrainer, GRPOConfig

fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")
tokenizer = AutoTokenizer.from_pretrained("./learner-full")

# --- Probe reward model to understand its output format ---
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "CraneAILabs/luganda-reward-model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
reward_tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/luganda-reward-model")

print(f"Reward model num_labels: {reward_model.config.num_labels}")
print(f"id2label: {getattr(reward_model.config, 'id2label', 'not set')}")

test_enc = reward_tokenizer("Test input in Luganda", return_tensors="pt")
test_enc = {k: v.to(reward_model.device) for k, v in test_enc.items()}
with torch.no_grad():
    test_out = reward_model(**test_enc)
print(f"Reward output shape: {test_out.logits.shape}")
print(f"Reward output values: {test_out.logits}")


def compute_rewards(completions, **kwargs):
    """
    Handles both binary classifier (2 logits → P(positive)) and regression (1 logit).
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(reward_model.device)

        with torch.no_grad():
            logits = reward_model(**inputs).logits

        if logits.shape[-1] == 2:
            score = torch.softmax(logits, dim=-1)[0, 1].item()
        else:
            score = logits[0, 0].item()

        rewards.append(score)
    return rewards


# --- Prepare GRPO prompts (user-turn only, not full chat text) ---
grpo_prompts = []
for item in fln['train'].select(range(min(300, len(fln['train'])))):
    text = item['text']
    if '<start_of_turn>user' in text and '<end_of_turn>' in text:
        user_msg = text.split('<start_of_turn>user\n')[1].split('<end_of_turn>')[0]
        grpo_prompts.append({"prompt": user_msg})

grpo_dataset = Dataset.from_list(grpo_prompts)
print(f"GRPO prompts: {len(grpo_dataset)}")

policy_model = AutoModelForCausalLM.from_pretrained(
    "./learner-full",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

grpo_trainer = GRPOTrainer(
    model=policy_model,
    reward_funcs=compute_rewards,
    args=GRPOConfig(
        output_dir="./grpo-output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        bf16=True,
        num_generations=4,
        max_new_tokens=200,
        max_prompt_length=512,
        logging_steps=10,
        save_steps=200,
        max_steps=600,
    ),
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
)

grpo_trainer.train()

grpo_trainer.model.save_pretrained("./grpo-full")
tokenizer.save_pretrained("./grpo-full")
print("Saved GRPO model to ./grpo-full")
