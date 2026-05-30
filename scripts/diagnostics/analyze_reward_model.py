"""
Analyzes reward model score distributions before and after GRPO.
Detects reward hacking and validates the reward signal.

Usage:
  python scripts/diagnostics/analyze_reward_model.py
"""
import json
import os
import statistics
from collections import Counter


def score_texts(texts, reward_model, reward_tokenizer):
    import torch

    scores = []
    for text in texts:
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(reward_model.device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits
        if logits.shape[-1] == 2:
            score = torch.softmax(logits, dim=-1)[0, 1].item()
        else:
            score = logits[0, 0].item()
        scores.append(score)
    return scores


def generate_completions(model, tokenizer, prompts, max_new_tokens=100):
    import torch

    completions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(text)
    return completions


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
    from scripts.core.data import load_fln_dataset, extract_first_letter

    os.makedirs("results/diagnostics", exist_ok=True)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "CraneAILabs/luganda-reward-model", torch_dtype=torch.bfloat16, device_map="auto"
    )
    reward_tok = AutoTokenizer.from_pretrained("CraneAILabs/luganda-reward-model")

    fln = load_fln_dataset("all")
    n = min(150, len(fln))
    sample_texts = [fln[i]["text"] for i in range(n)]
    sample_prompts = []
    for item in fln.select(range(n)):
        text = item["text"]
        if "<start_of_turn>user" in text and "<end_of_turn>" in text:
            sample_prompts.append(text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0])

    print("Scoring training data...")
    train_scores = score_texts(sample_texts, reward_model, reward_tok)
    results = {"train_data_scores": train_scores}

    for ckpt_name, ckpt_path in [("learner", "./learner-full"), ("grpo", "./grpo-full")]:
        if not os.path.isdir(ckpt_path):
            continue
        print(f"Generating and scoring {ckpt_name} completions...")
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, device_map="auto")
        tok = AutoTokenizer.from_pretrained(ckpt_path)
        completions = generate_completions(model, tok, sample_prompts[:100])
        del model
        torch.cuda.empty_cache()
        scores = score_texts(completions, reward_model, reward_tok)
        results[f"{ckpt_name}_scores"] = scores
        results[f"{ckpt_name}_letter_dist"] = dict(Counter(
            extract_first_letter(c) or "?" for c in completions
        ))

    with open("results/diagnostics/reward_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, label in [("train_data_scores", "Training data"),
                            ("learner_scores", "LEARNER outputs"),
                            ("grpo_scores", "GRPO outputs")]:
            if key in results:
                ax.hist(results[key], bins=20, alpha=0.5, label=label)
        ax.set_xlabel("Reward score")
        ax.set_ylabel("Count")
        ax.set_title("Reward Model Score Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig("results/diagnostics/reward_distribution.png", dpi=150)
        plt.close()
        print("Saved results/diagnostics/reward_distribution.png")
    except ImportError:
        pass

    print("\n--- Reward Score Summary ---")
    for key in ["train_data_scores", "learner_scores", "grpo_scores"]:
        if key in results:
            s = results[key]
            print(f"{key:<25} mean={statistics.mean(s):.3f}  std={statistics.stdev(s):.3f}")
    for key in ["learner_letter_dist", "grpo_letter_dist"]:
        if key in results:
            print(f"{key}: {results[key]}")
    print("\nSaved results/diagnostics/reward_analysis.json")


if __name__ == "__main__":
    main()
