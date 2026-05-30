"""
Reads all results/ablations/<axis>/<value>/results.json and produces:
  - A summary table printed to stdout
  - Per-axis plots saved to results/ablations/<axis>_plot.png
  - results/ablations/summary.json with best checkpoint info
Run after all ablation runs are complete.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

ABLATIONS_DIR = Path("results/ablations")
CONTROL = {
    "lora_rank": "16",
    "sft_epochs": "3",
    "balance_strategy": "oversample",
    "grpo_steps": "600",
}


def load_all_results():
    data = defaultdict(dict)
    for result_file in ABLATIONS_DIR.glob("*/*/results.json"):
        axis = result_file.parent.parent.name
        value = result_file.parent.name
        with open(result_file) as f:
            data[axis][value] = json.load(f)
    return data


def print_summary_table(data):
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    for axis, results in sorted(data.items()):
        print(f"\nAxis: {axis}  (control = {CONTROL.get(axis, '?')})")
        print(f"  {'Value':<20} {'Acc':>6} {'95% CI':>16} {'Spread':>8}")
        print("  " + "-" * 54)
        for value, r in sorted(results.items()):
            acc = r["accuracy"] * 100
            lo = r.get("ci_lower", 0) * 100
            hi = r.get("ci_upper", 0) * 100
            spread = r["spread"]
            ctrl = " ← control" if value == CONTROL.get(axis) else ""
            print(f"  {value:<20} {acc:>5.1f}% [{lo:>4.1f}%–{hi:>4.1f}%] {spread:>6.1f}pp{ctrl}")


def plot_axis(axis, results):
    import matplotlib.pyplot as plt

    values = sorted(results.keys())
    accs = [results[v]["accuracy"] * 100 for v in values]
    spreads = [results[v]["spread"] for v in values]
    lo_errs = [accs[i] - results[v].get("ci_lower", results[v]["accuracy"]) * 100
               for i, v in enumerate(values)]
    hi_errs = [results[v].get("ci_upper", results[v]["accuracy"]) * 100 - accs[i]
               for i, v in enumerate(values)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Ablation: {axis}")

    x = range(len(values))
    ax1.bar(x, accs, yerr=[lo_errs, hi_errs], capsize=5, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(values)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Benchmark Accuracy")

    ax2.bar(x, spreads, color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(values)
    ax2.set_ylabel("Spread (pp)")
    ax2.set_title("Position Bias Spread")

    plt.tight_layout()
    out = ABLATIONS_DIR / f"{axis}_plot.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot: {out}")


def find_best(data):
    best_acc, best_info = -1, {}
    for axis, results in data.items():
        for value, r in results.items():
            if r["accuracy"] > best_acc:
                best_acc = r["accuracy"]
                best_info = {"axis": axis, "value": value,
                             "accuracy": r["accuracy"], "spread": r["spread"]}
    return best_info


def main():
    data = load_all_results()
    if not data:
        print("No ablation results found in results/ablations/. Run run_ablation.py first.")
        return

    print_summary_table(data)

    for axis, results in data.items():
        if len(results) > 1:
            plot_axis(axis, results)

    best = find_best(data)
    summary = {"best": best, "all": {ax: dict(res) for ax, res in data.items()}}
    with open(ABLATIONS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nBest: axis={best.get('axis')} value={best.get('value')} acc={best.get('accuracy', 0):.1%}")
    print("Saved results/ablations/summary.json")


if __name__ == "__main__":
    main()
