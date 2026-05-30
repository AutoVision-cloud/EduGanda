from scripts.diagnostics.tokenizer_fertility import compute_fertility


class _MockTokenizer:
    """Splits on spaces — each word = 1 token. Fertility should be 1.0."""
    def encode(self, text, add_special_tokens=False):
        return text.strip().split()


class _FragmentingTokenizer:
    """Splits every character — simulates a tokenizer that knows no Luganda words."""
    def encode(self, text, add_special_tokens=False):
        return list(text.replace(" ", ""))


def test_fertility_uniform_tokenizer():
    tok = _MockTokenizer()
    texts = ["hello world", "foo bar baz"]
    result = compute_fertility(tok, texts)
    assert result["mean_fertility"] == 1.0
    assert result["overall_fertility"] == 1.0


def test_fertility_fragmenting_tokenizer_is_high():
    tok = _FragmentingTokenizer()
    texts = ["hello world"]  # 2 words, 10 chars → fertility 5.0
    result = compute_fertility(tok, texts)
    assert result["overall_fertility"] == 5.0


def test_fertility_empty_texts_skipped():
    tok = _MockTokenizer()
    texts = ["", "   ", "hello world"]
    result = compute_fertility(tok, texts)
    assert result["mean_fertility"] == 1.0


def test_fertility_returns_expected_keys():
    tok = _MockTokenizer()
    result = compute_fertility(tok, ["hello world"])
    for key in ["mean_fertility", "median_fertility", "overall_fertility", "total_words", "total_tokens"]:
        assert key in result
