# scripts/core/data.py
import re
import random
from itertools import permutations as _permutations
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

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
    """
    Extract the answer letter from a model response.
    Handles both SFT-style 'Okuddamu: X' outputs and bare letter responses.
    """
    import re
    # SFT models output "Okuddamu: X" (Luganda for "Answer: X") — check first
    m = re.search(r'[Oo]kuddamu\s*:\s*([ABCD])', text)
    if m:
        return m.group(1).upper()
    # Fall back to first standalone A/B/C/D (handles base model outputs)
    m = re.search(r'\b([ABCD])\b', text.upper())
    return m.group(1) if m else None


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


def build_training_dataset(
    balance_strategy: str = "oversample",
    use_permutation_augmentation: bool = True,
    n_perms_per_item: int = 3,
) -> "Dataset":
    """
    Loads FLN + exercises, applies position balancing or permutation augmentation,
    returns a Dataset with a single 'text' column ready for SFTTrainer.

    use_permutation_augmentation=True is preferred over balance_strategy='oversample':
    it generates genuine positional variety rather than duplicating examples.
    When True, balance_strategy is still applied first then augmentation runs on top.
    """
    from datasets import Dataset, concatenate_datasets
    fln = load_fln_dataset("all")
    exercises = load_exercises_dataset()

    mcq_items = [x for x in fln if x["correct_letter"] in ["A", "B", "C", "D"]]
    non_mcq = [x for x in fln if x["correct_letter"] not in ["A", "B", "C", "D"]]

    if use_permutation_augmentation:
        augmented = augment_mcq_with_permutations(mcq_items, n_perms_per_item=n_perms_per_item)
    else:
        augmented = balance_by_position(mcq_items, strategy=balance_strategy)

    return concatenate_datasets([
        Dataset.from_list(augmented),
        Dataset.from_list(non_mcq),
        exercises.select_columns(["text"]),
    ])


_LETTERS = ["A", "B", "C", "D"]
_IDENTITY_PERM = (0, 1, 2, 3)


def _parse_options(user_turn: str) -> Optional[Dict[str, str]]:
    """
    Extracts {A: text, B: text, C: text, D: text} from a chat-formatted user turn.
    Returns None if any option is missing.
    """
    options = {}
    for letter in _LETTERS:
        match = re.search(
            rf'\({letter}\)\s*(.+?)(?=\s*\([ABCD]\)|\s*<end_of_turn>|$)',
            user_turn,
            re.DOTALL,
        )
        if match:
            options[letter] = match.group(1).strip()
    return options if len(options) == 4 else None


def permute_mcq_item(item: Dict, perm: Tuple[int, ...]) -> Optional[Dict]:
    """
    Applies answer-option permutation to one MCQ item.
    perm[i] = index of old option placed at new position i.
    Returns None if the item text cannot be parsed.
    """
    text = item.get("text", "")
    correct_letter = item.get("correct_letter", "")
    if correct_letter not in _LETTERS:
        return None

    parts = text.split("<start_of_turn>model\n", 1)
    if len(parts) != 2:
        return None
    user_turn, model_turn = parts

    options = _parse_options(user_turn)
    if options is None:
        return None

    old_options = [options[l] for l in _LETTERS]
    new_options = {_LETTERS[i]: old_options[perm[i]] for i in range(4)}

    # Which new position contains the originally-correct option?
    correct_idx = _LETTERS.index(correct_letter)
    new_correct_idx = list(perm).index(correct_idx)
    new_correct_letter = _LETTERS[new_correct_idx]

    # Rebuild options block inside user turn
    new_opts_block = "\n".join(f"({l}) {new_options[l]}" for l in _LETTERS)
    new_user_turn = re.sub(
        r"\(A\).*?(?=\s*<end_of_turn>)",
        new_opts_block + "\n",
        user_turn,
        flags=re.DOTALL,
    )

    # Replace the correct letter reference in the model turn (first occurrence only)
    new_model_turn = re.sub(
        rf"\b{re.escape(correct_letter)}\b",
        new_correct_letter,
        model_turn,
        count=1,
    )

    new_item = dict(item)
    new_item["text"] = new_user_turn + "<start_of_turn>model\n" + new_model_turn
    new_item["correct_letter"] = new_correct_letter
    return new_item


def augment_mcq_with_permutations(
    mcq_items: List[Dict],
    n_perms_per_item: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    For each MCQ item, generates up to n_perms_per_item additional permuted versions.
    Original items are always included. Items that cannot be parsed are included unchanged.
    More principled than oversampling: generates genuine positional variety without repeats.
    """
    rng = random.Random(seed)
    non_identity = [p for p in _permutations(range(4)) if p != _IDENTITY_PERM]

    result = []
    for item in mcq_items:
        result.append(item)
        if item.get("correct_letter") not in _LETTERS:
            continue
        for perm in rng.sample(non_identity, min(n_perms_per_item, len(non_identity))):
            augmented = permute_mcq_item(item, perm)
            if augmented is not None:
                result.append(augmented)
    return result


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
