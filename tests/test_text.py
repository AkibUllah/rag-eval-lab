from rag_eval_lab.text import (
    is_bangla,
    normalize,
    overlap_ratio,
    raw_tokens,
    sentence_spans,
    split_sentences,
    stem_bn,
    stem_en,
    tokenize,
)


def test_normalize_folds_bangla_digits_and_strips_zero_width():
    assert normalize("৳৫,০০০\u200c") == "৳5,000"
    assert normalize("কার্যদিবস\u200d") == "কার্যদিবস"


def test_bangla_words_are_not_split_at_vowel_signs():
    # \w alone would split টাকা at the vowel sign; the tokenizer must keep whole words.
    assert raw_tokens("টাকা তুলুন") == ["টাকা", "তুলুন"]


def test_digit_grouping_is_removed_for_western_and_south_asian_styles():
    assert raw_tokens("BDT 25,000 and ১,৫০,০০০ টাকা") == ["bdt", "25000", "and", "150000", "টাকা"]


def test_currency_sign_is_not_a_token():
    assert raw_tokens("৳500") == ["500"]


def test_stopwords_removed_but_cash_out_survives():
    assert tokenize("What is the cash out limit?") == ["cash", "out", "limit"]
    assert tokenize("ক্যাশ আউট চার্জ কত?") == ["ক্যাশ", "আউট", "চার্জ"]


def test_english_stemmer_rules():
    assert [
        stem_en(w) for w in ["policies", "charging", "deducted", "boxes", "fees", "access", "pin"]
    ] == [
        "policy",
        "charg",
        "deduct",
        "box",
        "fee",
        "access",
        "pin",
    ]


def test_bangla_suffix_stripping_keeps_short_stems():
    assert stem_bn("এজেন্টের") == "এজেন্ট"
    assert stem_bn("টাকায়") == "টাকা"
    assert stem_bn("কে") == "কে"  # nothing left to strip to


def test_sentence_split_handles_danda_lists_and_headings():
    text = "First. Second?\n\n## Heading\n- item one\n- item two। বাংলা বাক্য। শেষ"
    assert split_sentences(text) == [
        "First.",
        "Second?",
        "## Heading",
        "- item one",
        "- item two।",
        "বাংলা বাক্য।",
        "শেষ",
    ]


def test_sentence_spans_point_back_into_text():
    text = "Alpha one. Beta two? Gamma."
    spans = sentence_spans(text, offset=10)
    assert [text[s - 10 : e - 10] for s, e in spans] == ["Alpha one.", "Beta two?", "Gamma."]


def test_is_bangla_and_overlap_ratio():
    assert is_bangla("পিন রিসেট PIN")
    assert not is_bangla("PIN reset")
    assert overlap_ratio(["a", "b"], ["b", "c"]) == 0.5
    assert overlap_ratio([], ["x"]) == 0.0
