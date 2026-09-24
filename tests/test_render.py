"""CPU-only rendering checks. No OCR models or remote resources are needed."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from pypdf import PdfReader

from render import _crop_url, _review_html, _wrap, render_run


def result(text="A readable English line.", *, ocr="A readab1e English line.", status="done"):
    return {"id": "sample-1", "status": status, "size": [512, 768], "needs_review": [],
            "lines": [{"id": "line-1", "box": [0, 0, 500, 20], "order": 0, "column": 0,
                       "ocr_text": ocr, "text": text, "needs_review": [], "changes": [],
                       "candidates": [], "observations": []}]}


def pdf_text(path):
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


class RenderTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_layout_order_keeps_spanning_sections_before_their_columns(self):
        expected = ["TOP HEADING", "UPPER RIGHT", "MID SECTION", "LOWER LEFT", "LOWER RIGHT"]
        columns = [-1, 1, -1, 0, 1]
        page = result()
        page["lines"] = [dict(page["lines"][0], id=f"line-{index}", order=index,
                              column=column, text=text, ocr_text=text)
                         for index, (column, text) in enumerate(zip(columns, expected))]
        page["lines"].reverse()  # Input order must not hide a sort regression.
        report = render_run([{"id": "sample-1"}], [page], self.tmp_path)
        for key in ("ocr_pdf", "recovered_pdf", "comparison_pdf"):
            text = pdf_text(report[key])
            positions = [text.index(label) for label in expected]
            self.assertEqual(positions, sorted(positions))
        text = Path(report["review_html"]).read_text(encoding="utf-8")
        positions = [text.index(label) for label in expected]
        self.assertEqual(positions, sorted(positions))

    def test_overflow_and_long_words_are_preserved(self):
        tmp_path = self.tmp_path
        text = "START " + "abcdefghij" * 1500 + " END <&> \\ [brackets] {braces} ` ~ ! # % ^ |"
        report = render_run([{"id": "sample-1"}], [result(text)], tmp_path)
        recovered = pdf_text(report["recovered_pdf"])
        comparison = pdf_text(report["comparison_pdf"])
        assert report["pages"]["recovered.pdf"] > 1
        assert report["pages"]["recovered.pdf"] == report["pages"]["comparison.pdf"]
        for extracted in (recovered, comparison):
            normalized = " ".join(extracted.split())
            assert "START" in normalized and "END <&> \\ [brackets] {braces} ` ~ ! # % ^ |" in normalized
            # Body character counts detect truncation even across hard word breaks.
            assert extracted.count("j") == 1500
        assert "A readab1e English line." in pdf_text(report["ocr_pdf"])
        assert "START" not in pdf_text(report["ocr_pdf"])

    def test_no_clear_is_explicit_and_reference_never_changes_text(self):
        tmp_path = self.tmp_path
        image = tmp_path / "out.png"
        Image.new("RGB", (50, 50), "white").save(image)
        samples = [{"id": "sample-1", "out": str(image)}]
        report = render_run(samples, [result()], tmp_path / "run")
        assert "[Clear unavailable]" in pdf_text(report["comparison_pdf"])
        assert {"id": "sample-1", "kind": "Clear"} in report["missing_images"]
        with_ref = render_run(samples, [result()], tmp_path / "referenced", {"sample-1": str(image)})
        assert pdf_text(with_ref["recovered_pdf"]) == pdf_text(report["recovered_pdf"])
        assert pdf_text(with_ref["ocr_pdf"]) == pdf_text(report["ocr_pdf"])

    def test_empty_and_error_pages_remain_visible(self):
        tmp_path = self.tmp_path
        empty = {"id": "sample-1", "status": "error", "error": "private/server/path", "lines": []}
        report = render_run([{"id": "sample-1"}, {"id": "missing"}], [empty], tmp_path)
        text = pdf_text(report["recovered_pdf"])
        assert report["pages"]["recovered.pdf"] == 2
        assert "Processing failed" in text and "No text lines" in text and "missing" in text
        assert "private/server/path" not in text
        assert "private/server/path" not in Path(report["review_html"]).read_text(encoding="utf-8")

    def test_empty_run_and_orphan_result_are_not_dropped(self):
        tmp_path = self.tmp_path
        report = render_run([], [], tmp_path / "empty")
        assert report["pages"]["comparison.pdf"] == 1
        assert "No samples" in pdf_text(report["recovered_pdf"])
        report = render_run([], [result()], tmp_path / "orphan")
        assert "A readable English line." in pdf_text(report["recovered_pdf"])

    def test_html_escapes_all_text_and_rejects_external_images(self):
        tmp_path = self.tmp_path
        payload = '<script>alert("x")</script><img src=x onerror=alert(1)>'
        row = result(payload)
        row["needs_review"] = [payload]
        line = row["lines"][0]
        line["changes"] = [{"before": payload, "after": payload}]
        line["observations"] = [{"model": payload, "variant": payload, "raw_text": payload,
                                  "text": payload, "crop": "https://example.com/leak.png"},
                                 {"crop": "../secret.png"}, {"crop": "javascript:alert(1)"}]
        output = _review_html([({"id": payload}, row)], tmp_path)
        assert "<script>" not in output and "<img src=x" not in output
        assert "&lt;script&gt;" in output and "&quot;x&quot;" in output
        assert 'src="https:' not in output and "../secret.png" not in output
        assert "Content-Security-Policy" in output

    def test_crop_paths_are_local_rasters_and_url_encoded(self):
        tmp_path = self.tmp_path
        crop = tmp_path / "crop #&'.png"
        Image.new("RGB", (10, 10), "white").save(crop)
        assert _crop_url(crop.name, tmp_path) == "crop%20%23%26%27.png"
        assert _crop_url(str(crop), tmp_path) is None
        assert _crop_url("../outside.png", tmp_path) is None
        (tmp_path / "untrusted.svg").write_text("<svg/>")
        assert _crop_url("untrusted.svg", tmp_path) is None

    def test_wrap_preserves_ascii_punctuation_and_long_tokens(self):
        token = "abc<>[]{}\\`~!@#$%^&*()-_=+|;:'\",./?" * 20
        rows = _wrap(token, 120, 10.5)
        assert "".join(rows) == token
        assert len(rows) > 1


if __name__ == "__main__":
    unittest.main()
