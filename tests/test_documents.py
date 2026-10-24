import pytest

from rag_eval_lab.documents import (
    LoaderError,
    load_bytes,
    load_directory,
    make_document,
    parse_front_matter,
)


def _pdf_with_text(text: str) -> bytes:
    """A minimal one-page PDF with a text layer, built by hand with a correct xref."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def test_front_matter_is_parsed_and_removed():
    meta, body = parse_front_matter("---\ntitle: Fees\nlang: en\n---\n# Fees\nBody")
    assert meta == {"title": "Fees", "lang": "en"}
    assert body == "# Fees\nBody"


def test_title_falls_back_to_first_heading_and_lang_is_detected():
    doc = make_document("# পিন রিসেট\n\nপিন ভুলে গেলে অ্যাপে রিসেট করুন।", doc_id="pin")
    assert doc.title == "পিন রিসেট"
    assert doc.lang == "bn"


def test_empty_document_is_rejected():
    with pytest.raises(LoaderError):
        make_document("---\ntitle: x\n---\n\n", doc_id="x")


def test_unsupported_extension_and_bad_utf8():
    with pytest.raises(LoaderError, match="unsupported"):
        load_bytes("sheet.xlsx", b"data")
    with pytest.raises(LoaderError, match="UTF-8"):
        load_bytes("notes.txt", b"\xff\xfe\xfa")


def test_pdf_text_layer_is_extracted():
    doc = load_bytes("Refund Policy.pdf", _pdf_with_text("Refunds are issued within 7 days."))
    assert doc.id == "refund-policy"
    assert "Refunds are issued within 7 days." in doc.text


def test_bundled_corpus_loads_with_unique_ids(corpus):
    assert len(corpus) == 29
    assert sum(1 for d in corpus if d.lang == "bn") == 5
    assert len({d.id for d in corpus}) == len(corpus)


def test_duplicate_ids_in_directory_are_rejected(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "fees.md").write_text("# Fees\nOne")
    (tmp_path / "fees.txt").write_text("Fees two")
    with pytest.raises(LoaderError, match="duplicate"):
        load_directory(tmp_path)
