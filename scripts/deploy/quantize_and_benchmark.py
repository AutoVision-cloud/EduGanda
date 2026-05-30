"""
Quantizes the best checkpoint to GGUF and benchmarks inference on-device.

Prerequisites:
  pip install llama-cpp-python
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
  export LLAMA_CPP_DIR=/path/to/llama.cpp

Usage:
  python scripts/deploy/quantize_and_benchmark.py [--model-path ./learner-full]
"""
import argparse
import json
import os
import subprocess
import time

QUANT_LEVELS = ["Q4_K_M", "Q8_0"]


def convert_to_gguf(model_path: str, output_dir: str, llama_cpp_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "model-f16.gguf")
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    subprocess.run(
        ["python3", convert_script, model_path, "--outfile", out_file, "--outtype", "f16"],
        check=True,
    )
    print(f"Converted to GGUF: {out_file}")
    return out_file


def quantize_gguf(gguf_f16_path: str, output_dir: str, quant: str, llama_cpp_dir: str) -> str:
    quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")
    out_file = os.path.join(output_dir, f"model-{quant}.gguf")
    subprocess.run([quantize_bin, gguf_f16_path, out_file, quant], check=True)
    print(f"Quantized {quant}: {out_file}")
    return out_file


def benchmark_gguf(gguf_path: str, benchmark_ds, n_items: int = 100) -> dict:
    from llama_cpp import Llama
    from scripts.core.data import extract_first_letter

    file_size_mb = os.path.getsize(gguf_path) / 1024 / 1024
    llm = Llama(model_path=gguf_path, n_ctx=1024, n_gpu_layers=-1, verbose=False)

    correct = 0
    total_tokens = 0
    start = time.time()

    for item in list(benchmark_ds["train"])[:n_items]:
        prompt = (
            f"<start_of_turn>user\nAnswer with only the letter (A, B, C, or D). Do not explain.\n\n"
            f"{item['luganda_question']}\n"
            f"(A) {item['luganda_answer_a']}\n(B) {item['luganda_answer_b']}\n"
            f"(C) {item['luganda_answer_c']}\n(D) {item['luganda_answer_d']}\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        out = llm(prompt, max_tokens=5, temperature=0.01, repeat_penalty=1.2)
        response = out["choices"][0]["text"]
        if extract_first_letter(response) == item["correct_answer"]:
            correct += 1
        total_tokens += out["usage"]["completion_tokens"]

    elapsed = time.time() - start
    return {
        "gguf_path": gguf_path,
        "file_size_mb": round(file_size_mb, 1),
        "accuracy": correct / n_items,
        "tokens_per_sec": round(total_tokens / elapsed, 1),
        "n_items": n_items,
    }


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="HF checkpoint path. Defaults to best ablation or ./learner-full.")
    args = parser.parse_args()

    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "")
    if not llama_cpp_dir or not os.path.isdir(llama_cpp_dir):
        print("ERROR: Set LLAMA_CPP_DIR env var to your llama.cpp build directory.")
        print("  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
        raise SystemExit(1)

    model_path = args.model_path
    if not model_path:
        summary_path = "results/ablations/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            best = summary.get("best", {})
            axis, value = best.get("axis"), best.get("value")
            candidate = f"./ablation-{axis}-{value}-sft" if axis and value else None
            if candidate and os.path.isdir(candidate):
                model_path = candidate
                print(f"Using best ablation checkpoint: {model_path}")
        if not model_path:
            model_path = "./learner-full"
            print(f"Using default checkpoint: {model_path}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    os.makedirs("results/deploy", exist_ok=True)
    gguf_dir = "results/deploy/gguf"
    gguf_f16 = convert_to_gguf(model_path, gguf_dir, llama_cpp_dir)

    benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")
    benchmark_results = []

    for quant in QUANT_LEVELS:
        gguf_path = quantize_gguf(gguf_f16, gguf_dir, quant, llama_cpp_dir)
        print(f"\nBenchmarking {quant}...")
        result = benchmark_gguf(gguf_path, benchmark, n_items=100)
        result["quant"] = quant
        benchmark_results.append(result)
        print(f"  accuracy={result['accuracy']:.1%}  speed={result['tokens_per_sec']:.1f} tok/s  size={result['file_size_mb']:.0f}MB")

    with open("results/deploy/quantization_benchmark.json", "w") as f:
        json.dump({"source_model": model_path, "results": benchmark_results}, f, indent=2)

    print("\n--- Quantization Benchmark ---")
    print(f"{'Quant':<10} {'Accuracy':>9} {'Tok/s':>8} {'Size (MB)':>10}")
    print("-" * 42)
    for r in benchmark_results:
        print(f"{r['quant']:<10} {r['accuracy']:>8.1%} {r['tokens_per_sec']:>8.1f} {r['file_size_mb']:>9.0f}")

    print("\nSaved results/deploy/quantization_benchmark.json")


if __name__ == "__main__":
    main()
