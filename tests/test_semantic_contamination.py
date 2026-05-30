from scripts.diagnostics.semantic_contamination import find_overlaps


def test_identical_texts_flagged():
    bench = ["Omusomesa ayagala okusoma"]
    train = ["Omusomesa ayagala okusoma"]
    overlaps = find_overlaps(bench, train, threshold=0.9)
    assert len(overlaps) == 1
    assert overlaps[0]["similarity"] > 0.9


def test_unrelated_texts_not_flagged():
    bench = ["The quick brown fox jumps"]
    train = ["Luganda language literacy education"]
    overlaps = find_overlaps(bench, train, threshold=0.5)
    assert len(overlaps) == 0


def test_partial_overlap_detected():
    bench = ["foundational literacy numeracy Uganda schools"]
    train = ["foundational literacy numeracy Uganda primary"]
    overlaps = find_overlaps(bench, train, threshold=0.5)
    assert len(overlaps) == 1


def test_results_sorted_by_similarity_descending():
    bench = ["abc def ghi", "xyz uvw rst"]
    train = ["abc def ghi jkl", "completely different text here"]
    overlaps = find_overlaps(bench, train, threshold=0.0)
    sims = [o["similarity"] for o in overlaps]
    assert sims == sorted(sims, reverse=True)


def test_returns_expected_keys():
    bench = ["hello world"]
    train = ["hello world"]
    overlaps = find_overlaps(bench, train, threshold=0.5)
    assert len(overlaps) > 0
    for key in ["benchmark_idx", "train_idx", "similarity", "benchmark_text", "train_text"]:
        assert key in overlaps[0]
