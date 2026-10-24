import pytest

from rag_eval_lab.chunking import (
    CHUNKERS,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    chunk_documents,
    get_chunker,
    sections,
)
from rag_eval_lab.text import word_count


@pytest.mark.parametrize("name", list(CHUNKERS))
def test_every_chunk_is_an_exact_span_of_its_document(corpus, name):
    by_id = {d.id: d for d in corpus}
    chunks = chunk_documents(corpus, get_chunker(name))
    assert chunks
    for c in chunks:
        assert by_id[c.doc_id].text[c.start : c.end] == c.text
        assert c.id == f"{c.doc_id}#{c.index}"
    # ids are unique and dense per document
    assert len({c.id for c in chunks}) == len(chunks)


def test_fixed_chunker_windows_overlap(small_doc):
    chunks = FixedSizeChunker(size=12, overlap=4).chunk(small_doc)
    assert len(chunks) > 2
    assert all(len(c.text.split()) <= 12 for c in chunks)
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-4:] == second_words[:4]


def test_sections_build_heading_paths(small_doc):
    secs = sections(small_doc)
    assert [s.heading for s in secs] == [
        "Cash out charges",
        "Cash out charges > Agents",
        "Cash out charges > ATM",
    ]


def test_recursive_chunks_stay_inside_one_section(small_doc):
    chunks = RecursiveChunker(max_words=40).chunk(small_doc)
    assert [c.heading for c in chunks] == [s.heading for s in sections(small_doc)]
    assert "1.39%" in chunks[1].text and "ATM" not in chunks[1].text


def test_recursive_splits_long_sections_by_sentence():
    from rag_eval_lab.documents import make_document

    body = " ".join(f"Sentence number {i} talks about fees." for i in range(30))
    doc = make_document(f"# Long\n\n{body}\n", doc_id="long")
    chunks = RecursiveChunker(max_words=40).chunk(doc)
    assert len(chunks) > 3
    assert all(word_count(c.text) <= 40 for c in chunks)


def test_sentence_window_indexes_one_sentence_but_returns_neighbours(small_doc):
    chunks = SentenceWindowChunker(window=1).chunk(small_doc)
    agent = next(c for c in chunks if c.search_text.endswith("Using USSD the charge is 1.49%."))
    assert "1.39%" in agent.text  # previous sentence is part of the window
    assert "1.39%" not in agent.search_text.splitlines()[-1]
    assert all(not c.text.startswith("#") for c in chunks)


def test_search_text_has_title_and_heading_but_no_markdown_heading_lines(small_doc):
    chunk = RecursiveChunker().chunk(small_doc)[1]
    lines = chunk.search_text.splitlines()
    assert lines[0] == "Cash out charges" and lines[1] == "Agents"
    assert not any(line.startswith("#") for line in lines)


def test_invalid_parameters_and_unknown_chunker():
    with pytest.raises(ValueError):
        FixedSizeChunker(size=10, overlap=10)
    with pytest.raises(ValueError):
        RecursiveChunker(max_words=5)
    with pytest.raises(ValueError, match="unknown chunker"):
        get_chunker("semantic")
