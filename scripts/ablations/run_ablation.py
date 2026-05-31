"""
Single ablation run. Usage:
  python scripts/ablations/run_ablation.py --axis lora_rank --value 32
  python scripts/ablations/run_ablation.py --axis sft_epochs --value 1
  python scripts/ablations/run_ablation.py --axis balance_strategy --value none
  python scripts/ablations/run_ablation.py --axis grpo_steps --value 200

Saves results to results/ablations/<axis>/<value>/results.json
"""
import argparse
import json
import os

VALID_AXES = {
    "lora_rank": ["8", "16", "32"],
    "sft_epochs": ["1", "3", "5"],
    "balance_strategy": ["none", "oversample", "undersample"],
    "grpo_steps": ["200", "600", "1000"],
}


def main():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.core.data import build_training_dataset, load_fln_dataset, build_grpo_prompts
    from scripts.core.train import train_sft, train_grpo
    from scripts.core.evaluate import evaluate_on_benchmark, bootstrap_ci
    from scripts.core.callbacks import DiagnosticCallback

    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=VALID_AXES.keys())
    parser.add_argument("--value", required=True)
    args = parser.parse_args()

    if args.value not in VALID_AXES[args.axis]:
        parser.error(f"--value must be one of {VALID_AXES[args.axis]} for axis '{args.axis}'")

    out_dir = f"results/ablations/{args.axis}/{args.value}"
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_dir = f"./ablation-{args.axis}-{args.value}"
    benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

    balance_strategy = "oversample"
    lora_rank = 16
    num_train_epochs = 3
    grpo_steps = 600

    if args.axis == "balance_strategy":
        balance_strategy = args.value
    elif args.axis == "lora_rank":
        lora_rank = int(args.value)
    elif args.axis == "sft_epochs":
        num_train_epochs = int(args.value)
    elif args.axis == "grpo_steps":
        grpo_steps = int(args.value)

    cb = DiagnosticCallback(f"{out_dir}/training_curves.jsonl")

    print(f"\n[Ablation] axis={args.axis} value={args.value}")
    train_dataset = build_training_dataset(balance_strategy=balance_strategy)
    sft_path = checkpoint_dir + "-sft"
    train_sft(
        model_path="CraneAILabs/ganda-gemma-1b",
        train_dataset=train_dataset,
        output_dir=sft_path,
        lora_rank=lora_rank,
        num_train_epochs=num_train_epochs,
        callbacks=[cb],
    )

    if args.axis == "grpo_steps":
        fln = load_fln_dataset("all")
        grpo_prompts = build_grpo_prompts(fln, n=300)
        grpo_path = checkpoint_dir + "-grpo"
        train_grpo(
            model_path=sft_path,
            grpo_dataset=grpo_prompts,
            output_dir=grpo_path,
            max_steps=grpo_steps,
            callbacks=[cb],
        )
        eval_path = grpo_path
    else:
        eval_path = sft_path

    model = AutoModelForCausalLM.from_pretrained(eval_path, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(eval_path)
    result = evaluate_on_benchmark(model, tok, benchmark, label=f"{args.axis}={args.value}")
    acc, lo, hi = bootstrap_ci(result["predictions"], result["labels"])
    result["ci_lower"] = lo
    result["ci_upper"] = hi
    result["axis"] = args.axis
    result["value"] = args.value
    del model
    torch.cuda.empty_cache()

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResult: accuracy={result['accuracy']:.1%} [{lo:.1%}–{hi:.1%}]  spread={result['spread']:.1f}pp")
    print(f"Saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
