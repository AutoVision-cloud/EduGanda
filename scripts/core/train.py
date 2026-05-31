# scripts/core/train.py
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import TrainerCallback


def train_sft(
    model_path: str,
    train_dataset,
    output_dir: str,
    lora_rank: int = 16,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 1024,
    report_to: str = "none",
    callbacks: Optional[List] = None,
) -> str:
    """Runs SFT with QLoRA, merges adapter, saves full model to output_dir. Returns output_dir."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir + "-adapter",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            max_seq_length=max_seq_length,
            report_to=report_to,
        ),
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        callbacks=callbacks or [],
    )

    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged SFT model to {output_dir}")
    return output_dir


def _build_reward_fn(reward_model_path: str):
    """Returns a GRPOTrainer-compatible reward function."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

    def compute_rewards(completions, **kwargs):
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

    return compute_rewards


def train_grpo(
    model_path: str,
    grpo_dataset,
    output_dir: str,
    reward_model_path: str = "CraneAILabs/luganda-reward-model",
    max_steps: int = 600,
    learning_rate: float = 5e-6,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_generations: int = 4,
    report_to: str = "none",
    callbacks: Optional[List] = None,
) -> str:
    """Runs GRPO on model_path using the reward model. Saves full model to output_dir."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    compute_rewards = _build_reward_fn(reward_model_path)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=compute_rewards,
        args=GRPOConfig(
            output_dir=output_dir + "-ckpt",
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=True,
            num_generations=num_generations,
            max_new_tokens=200,
            max_prompt_length=512,
            logging_steps=10,
            save_steps=200,
            max_steps=max_steps,
            report_to=report_to,
        ),
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        callbacks=callbacks or [],
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved GRPO model to {output_dir}")
    return output_dir
