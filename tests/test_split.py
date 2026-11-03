import dataclasses
import importlib.util
import json
import random

import pytest

from rag_eval_lab.config import PROJECT_ROOT
from rag_eval_lab.evaluation.dataset import (
    Evidence,
    GoldenItem,
    GoldenSetError,
    load_golden,
    select_split,
)
from rag_eval_lab.evaluation.split import (
    assign_splits,
    split_hash,
    split_summary,
    stratum,
    target_test_size,
)
from tests.conftest import GOLDEN


def _item(i: int, tags: list[str], answerable: bool = True, split: str | None = None):
    evidence = [Evidence("doc", "text")] if answerable else []
    return GoldenItem(f"t{i:03d}", f"question {i}", evidence, "", tags, split)


def _synthetic(n: int = 60) -> list[GoldenItem]:
    kinds = [["lexical"], ["paraphrase"], ["bn"], ["numeric"]]
    items = [_item(i, kinds[i % 4]) for i in range(n)]
    return items + [_item(n + j, ["unanswerable"], answerable=False) for j in range(4)]


def test_stratum_groups_by_answerability_language_and_wording():
    assert stratum(_item(1, ["unanswerable", "bn"], answerable=False)) == "unanswerable"
    assert stratum(_item(2, ["bn", "paraphrase"])) == "bn"
    assert stratum(_item(3, ["paraphrase", "numeric"])) == "en/paraphrase"
    assert stratum(_item(4, ["lexical", "multi-doc"])) == "en/multi-doc"
    assert stratum(_item(7, ["paraphrase", "multi-evidence"])) == "en/multi-evidence"
    assert stratum(_item(6, ["lexical", "numeric"])) == "en/lexical"
    assert stratum(_item(5, ["reasoning"])) == "en/other"


def test_target_test_size_rounds_half_up():
    assert [target_test_size(n, 0.3) for n in (1, 2, 4, 5, 10, 36)] == [0, 1, 1, 2, 3, 11]


def test_assignment_is_deterministic_and_order_independent():
    items = _synthetic()
    first = assign_splits(items)
    shuffled = items[:]
    random.Random(7).shuffle(shuffled)
    assert assign_splits(shuffled) == first
    assert set(first.values()) == {"dev", "test"}
    assert split_hash("q001") == split_hash("q001") != split_hash("q001", seed="other")
    assert assign_splits(items, seed="other") != first


def test_assignment_hits_the_target_share_in_every_stratum():
    items = _synthetic()
    splits = assign_splits(items)
    by_stratum: dict[str, list[str]] = {}
    for item in items:
        by_stratum.setdefault(stratum(item), []).append(splits[item.id])
    for members in by_stratum.values():
        assert members.count("test") == target_test_size(len(members), 0.3)


def test_adding_questions_never_moves_existing_ones():
    items = _synthetic()
    splits = assign_splits(items)
    stored = [dataclasses.replace(i, split=splits[i.id]) for i in items]
    grown = stored + [_item(100 + j, ["lexical"]) for j in range(10)]
    regrown = assign_splits(grown)
    assert {i.id: regrown[i.id] for i in stored} == splits
    lexical = [i for i in grown if stratum(i) == "en/lexical"]
    assert [regrown[i.id] for i in lexical].count("test") == target_test_size(len(lexical), 0.3)


def test_stored_splits_win_unless_reassigned():
    items = [_item(i, ["lexical"], split="test") for i in range(10)]
    assert set(assign_splits(items).values()) == {"test"}
    fresh = assign_splits(items, keep_existing=False)
    assert list(fresh.values()).count("test") == 3


def test_assignment_rejects_bad_fraction():
    for fraction in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="test_fraction"):
            assign_splits(_synthetic(), test_fraction=fraction)


def test_split_summary_counts_strata_languages_and_tags():
    items = [_item(1, ["bn"], split="dev"), _item(2, ["lexical"], split="test"), _item(3, [])]
    summary = split_summary(items)
    assert summary["all"] == {"dev": 1, "test": 1, "unassigned": 1}
    assert summary["lang:bn"] == {"dev": 1} and summary["tag:lexical"] == {"test": 1}
    assert summary["stratum:en/other"] == {"unassigned": 1}


def test_load_golden_reads_and_validates_split(tmp_path):
    path = tmp_path / "g.jsonl"
    rows = [
        {"id": "a", "question": "q", "split": "dev"},
        {"id": "b", "question": "q", "split": "test"},
        {"id": "c", "question": "q"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert [i.split for i in load_golden(path)] == ["dev", "test", None]
    path.write_text(json.dumps({"id": "a", "question": "q", "split": "holdout"}) + "\n")
    with pytest.raises(GoldenSetError, match="unknown split 'holdout'"):
        load_golden(path)


def test_select_split_filters_and_fails_loudly():
    items = [_item(1, [], split="dev"), _item(2, [], split="test"), _item(3, [], split="dev")]
    assert [i.id for i in select_split(items, "dev")] == ["t001", "t003"]
    assert [i.id for i in select_split(items, "test")] == ["t002"]
    assert select_split(items, "all") == items
    with pytest.raises(GoldenSetError, match="unknown split 'train'"):
        select_split(items, "train")
    unassigned = items + [_item(4, [])]
    assert len(select_split(unassigned, "all")) == 4
    with pytest.raises(GoldenSetError, match="1 golden items have no split"):
        select_split(unassigned, "test")


def _split_script():
    path = PROJECT_ROOT / "scripts" / "split_golden.py"
    spec = importlib.util.spec_from_file_location("split_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_split_is_reproducible(golden):
    stored = {i.id: i.split for i in golden}
    assert assign_splits(golden, keep_existing=False) == stored


def test_bundled_split_is_balanced_across_languages_and_tags(golden):
    summary = split_summary(golden)
    assert summary["all"] == {"dev": 49, "test": 21}
    for key in ("lang:en", "lang:bn", "tag:lexical", "tag:paraphrase", "tag:numeric"):
        counts = summary[key]
        assert 0.2 <= counts["test"] / (counts["dev"] + counts["test"]) <= 0.4, key
    for key, counts in summary.items():
        assert counts["dev"] >= 1 and counts["test"] >= 1, key


def test_split_script_assigns_new_questions_and_checks(tmp_path, capsys):
    script = _split_script()
    path = tmp_path / "golden.jsonl"
    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    new = {"id": "q999", "question": "new?", "relevant": [], "tags": ["unanswerable"]}
    path.write_text("\n".join([*lines, json.dumps(new)]) + "\n", encoding="utf-8")
    assert script.main(["--golden", str(path), "--check"]) == 1
    assert "1 questions have no split: q999" in capsys.readouterr().out
    assert script.main(["--golden", str(path)]) == 0
    assert "1 of 71 questions assigned" in capsys.readouterr().out
    rewritten = path.read_text(encoding="utf-8").splitlines()
    assert rewritten[:70] == lines  # existing questions untouched, byte for byte
    assert json.loads(rewritten[70])["split"] in ("dev", "test")
    assert script.main(["--golden", str(path), "--check"]) == 0
