# scripts/core/data.py
import random
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


def balance_by_position(
    mcq_items: List[Dict],
    strategy: str = "oversample",
    seed: int = 42,
) -> List[Dict]:
    """
    strategy: 'oversample' | 'undersample' | 'none'
    Returns a new list (does not mutate input).
    """
    if strategy == "none":
        return list(mcq_items)

    rng = random.Random(seed)
    by_pos = {}
    for item in mcq_items:
        by_pos.setdefault(item["correct_letter"], []).append(item)

    if strategy == "oversample":
        target = max(len(v) for v in by_pos.values())
        result = []
        for items in by_pos.values():
            if len(items) < target:
                result.extend(items * (target // len(items)))
                result.extend(rng.sample(items, target % len(items)))
            else:
                result.extend(items)
    elif strategy == "undersample":
        target = min(len(v) for v in by_pos.values())
        result = []
        for items in by_pos.values():
            result.extend(rng.sample(items, target))
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'oversample', 'undersample', or 'none'.")

    rng.shuffle(result)
    return result


def extract_first_letter(text: str) -> Optional[str]:
    """Extract the first standalone A/B/C/D letter from a model response string."""
    import re
    match = re.search(r'\b([ABCD])\b', text.upper())
    return match.group(1) if match else None


def load_fln_dataset(subset: str = "all") -> "Dataset":
    from datasets import load_dataset
    return load_dataset("CraneAILabs/luganda-fln-training-data", subset)["train"]


def load_exercises_dataset() -> "Dataset":
    from datasets import load_dataset
    ds = load_dataset("CraneAILabs/luganda-bilingual-literacy-exercises")["train"]
    if "text" in ds.column_names:
        return ds

    def _format(row):
        cols = ds.column_names
        if "question" in cols and "answer" in cols:
            text = (
                f"<start_of_turn>user\n{row['question']}<end_of_turn>\n"
                f"<start_of_turn>model\n{row['answer']}<end_of_turn>"
            )
        elif "instruction" in cols and "output" in cols:
            text = (
                f"<start_of_turn>user\n{row['instruction']}<end_of_turn>\n"
                f"<start_of_turn>model\n{row['output']}<end_of_turn>"
            )
        else:
            content = " | ".join(str(row[c]) for c in cols if isinstance(row[c], str))
            text = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n<end_of_turn>"
        return {"text": text}

    return ds.map(_format)


def build_training_dataset(balance_strategy: str = "oversample") -> "Dataset":
    """
    Loads FLN + exercises, applies position balancing, returns a Dataset
    with a single 'text' column ready for SFTTrainer.
    """
    from datasets import Dataset, concatenate_datasets
    fln = load_fln_dataset("all")
    exercises = load_exercises_dataset()

    mcq_items = [x for x in fln if x["correct_letter"] in ["A", "B", "C", "D"]]
    non_mcq = [x for x in fln if x["correct_letter"] not in ["A", "B", "C", "D"]]

    balanced = balance_by_position(mcq_items, strategy=balance_strategy)

    return concatenate_datasets([
        Dataset.from_list(balanced),
        Dataset.from_list(non_mcq),
        exercises.select_columns(["text"]),
    ])


def build_grpo_prompts(fln_dataset: "Dataset", n: int = 300) -> "Dataset":
    """Extract user-turn prompts from pre-formatted FLN chat text."""
    prompts = []
    for item in fln_dataset.select(range(min(n, len(fln_dataset)))):
        text = item["text"]
        if "<start_of_turn>user" in text and "<end_of_turn>" in text:
            user_msg = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0]
            prompts.append({"prompt": user_msg})
    from datasets import Dataset
    return Dataset.from_list(prompts)
