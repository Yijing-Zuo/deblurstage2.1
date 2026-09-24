"""Render saved OCR results as readable PDFs and a static review page."""

from __future__ import annotations

import html
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from storage import normalize_typography


RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _text(value) -> str:
    return "" if value is None else str(value)


def _pdf_text(value) -> str:
    """Our transcription alphabet is ASCII; never send control codes to a font."""
    text = normalize_typography(_text(value))
    return "".join(c if c in "\n\t" or 32 <= ord(c) < 127 else "?" for c in text)


def _wrap(text: str, width: float, size: float) -> list[str]:
    """Wrap at spaces, and split oversized words without dropping characters."""
    rows = []
    for paragraph in _pdf_text(text).split("\n"):
        current = ""
        for word in paragraph.split():
            if current and stringWidth(current + " " + word, "Helvetica", size) <= width:
                current += " " + word
                continue
            if current:
                rows.append(current)
                current = ""
            for char in word:
                if current and stringWidth(current + char, "Helvetica", size) > width:
                    rows.append(current)
                    current = ""
                current += char
        rows.append(current)
    return rows


def _layout(result: dict, field: str, width: float, rows_per_page: int, size: float):
    rows = []
    if result.get("status") == "error":
        rows += _wrap("[Processing failed; available text is shown below.]", width, size)
    lines = sorted(result.get("lines", []), key=lambda line: (
        line.get("order", 0), line.get("column", 0), _text(line.get("id"))))
    for line in lines:
        text = _text(line.get(field))
        rows += _wrap(text if text.strip() else "[Unreadable line]", width, size)
    if not lines:
        rows += _wrap("[No text lines were recovered.]", width, size)
    return [rows[i:i + rows_per_page] for i in range(0, len(rows), rows_per_page)]


def _draw_text_page(pdf, rows, sample_id, label, part, total, page_size, margin, size, leading,
                    offset=0, top=0):
    width, height = page_size
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 10)
    heading = f"{label} | {_pdf_text(sample_id)} | {part}/{total}"
    # IDs are metadata, so wrapping them must not consume body space.
    pdf.drawString(offset + margin, top + height - margin, _wrap(heading, width - 2 * margin, 10)[0])
    text = pdf.beginText(offset + margin, top + height - margin - 25)
    text.setFont("Helvetica", size)
    text.setLeading(leading)
    for row in rows:
        text.textLine(row)
    pdf.drawText(text)


def _draw_image(pdf, value, label, rect) -> bool:
    """Read local raster images only. Error labels never expose server paths."""
    x, y, width, height = rect
    try:
        path = Path(_text(value))
        if not value or not path.is_absolute() or path.suffix.lower() not in RASTER_SUFFIXES:
            raise ValueError("Not a local raster image")
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
        try:
            pdf.drawImage(ImageReader(image), x, y, width, height,
                          preserveAspectRatio=True, anchor="c", mask="auto")
        finally:
            image.close()
        return True
    except Exception:
        pdf.setFont("Helvetica", 12)
        pdf.setFillColorRGB(0.4, 0.4, 0.4)
        pdf.drawCentredString(x + width / 2, y + height / 2, f"[{label} unavailable]")
        pdf.setFillColorRGB(0, 0, 0)
        return False


def _crop_url(value, output_dir: Path) -> str | None:
    """A review file may link only to raster assets inside its own run directory."""
    try:
        if not value or Path(_text(value)).is_absolute():
            return None
        path = (output_dir / _text(value)).resolve()
        relative = path.relative_to(output_dir.resolve())
        if path.suffix.lower() not in RASTER_SUFFIXES or not path.is_file():
            return None
        return quote(relative.as_posix(), safe="/")
    except (OSError, ValueError, RuntimeError):
        return None


def _escaped(value) -> str:
    return html.escape(_text(value), quote=True)


def _review_html(records: list[tuple[dict, dict]], output_dir: Path) -> str:
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>OCR line review</title><style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#171717;background:white}
h1,h2,h3{line-height:1.2}section{border-top:2px solid #bbb;margin-top:32px;padding-top:12px}
article{border:1px solid #ccc;padding:16px;margin:18px 0;overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.5 ui-monospace,monospace}
table{border-collapse:collapse;width:100%;table-layout:fixed}th,td{text-align:left;vertical-align:top;border:1px solid #ddd;padding:8px;overflow-wrap:anywhere}
th:first-child{width:22%}figure{margin:12px 0}img{display:block;max-width:100%;height:auto;border:1px solid #ddd}
.review{color:#764600}.muted{color:#555}nav a{margin-right:12px;display:inline-block}
</style><h1>OCR line review</h1>
<p>Images and model readings are evidence, not ground truth. Clear references appear only in the comparison PDF.</p><nav>''']
    for index, (sample, _) in enumerate(records):
        parts.append(f'<a href="#sample-{index}">{_escaped(sample["id"])}</a>')
    parts.append("</nav>")
    if not records:
        parts.append("<p>No samples were supplied.</p>")
    for index, (sample, result) in enumerate(records):
        parts.append(f'<section id="sample-{index}"><h2>{_escaped(sample["id"])}</h2>')
        parts.append(f'<p>Status: {_escaped(result.get("status", "error"))}</p>')
        if result.get("status") == "error":
            parts.append("<p class=review>Processing failed. Consult the run log for details.</p>")
        reasons = result.get("needs_review", [])
        if reasons:
            parts.append(f'<p class="review">Review: {_escaped("; ".join(map(str, reasons)))}</p>')
        lines = sorted(result.get("lines", []), key=lambda line: (
            line.get("order", 0), line.get("column", 0), _text(line.get("id"))))
        if not lines:
            parts.append("<p>[No text lines were recovered.]</p>")
        for line in lines:
            parts.append(f'<article><h3>{_escaped(line.get("id", "Line"))}</h3>')
            for label, key in (("Visual OCR", "ocr_text"), ("Final", "text")):
                parts.append(f'<strong>{label}</strong><pre>{_escaped(line.get(key))}</pre>')
            reasons = line.get("needs_review", [])
            parts.append(f'<p class="review">Review: {_escaped("; ".join(map(str, reasons))) if reasons else "None recorded"}</p>')
            changes = line.get("changes", [])
            parts.append(f'<p>Changes: {_escaped(changes) if changes else "None"}</p>')
            if line.get("candidates"):
                parts.append(f'<details><summary>Candidate evidence</summary>'
                             f'<pre>{_escaped(line["candidates"])}</pre></details>')
            observations = line.get("observations", [])
            crops = set()
            for observation in observations:
                url = _crop_url(observation.get("crop"), output_dir)
                if url and url not in crops:
                    crops.add(url)
                    parts.append(f'<figure><img src="{_escaped(url)}" alt="Line crop" loading="lazy">'
                                 f'<figcaption>{_escaped(observation.get("variant", "Crop"))}</figcaption></figure>')
            if not crops:
                parts.append("<p class=muted>Line crop unavailable.</p>")
            parts.append("<table><thead><tr><th>Model / variant</th><th>Original reading</th><th>Filtered reading</th></tr></thead><tbody>")
            for observation in observations:
                name = f'{_text(observation.get("model"))} / {_text(observation.get("variant"))}'
                parts.append(f'<tr><td>{_escaped(name)}</td><td>{_escaped(observation.get("raw_text"))}</td>'
                             f'<td>{_escaped(observation.get("text"))}</td></tr>')
            parts.append("</tbody></table></article>")
        parts.append("</section>")
    parts.append("</html>")
    return "".join(parts)


def render_run(samples: list, page_results: list, output_dir: Path,
               references: dict | None = None, settings: dict | None = None) -> dict:
    """Render existing results only; references are never used to change text.

    Each physical OCR line starts a new text line. Long lines wrap and overflow
    onto continuation pages. Files are fully generated before atomic replacement.
    """
    settings = settings or {}
    width, height = map(float, settings.get("page_size", (512, 768)))
    size = float(settings.get("font_size", 10.5))
    leading = float(settings.get("line_height", 15))
    margin = float(settings.get("margin", 28))
    if not all(math.isfinite(v) for v in (width, height, size, leading, margin)):
        raise ValueError("Render dimensions must be finite")
    if size <= 0 or leading < size or margin < 0 or width - 2 * margin < 100:
        raise ValueError("Invalid render dimensions")
    rows_per_page = int((height - 2 * margin - 25) // leading) + 1
    if rows_per_page < 2:
        raise ValueError("Page is too short for text")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    references = references or {}
    result_by_id = {row["id"]: row for row in page_results}
    sample_by_id = {sample["id"]: sample for sample in samples}
    if len(result_by_id) != len(page_results) or len(sample_by_id) != len(samples):
        raise ValueError("Duplicate sample IDs")
    # Preserve orphan result pages as well as samples with missing results.
    records = [(sample, result_by_id.get(sample["id"], {
        "id": sample["id"], "status": "error", "needs_review": ["missing_result"], "lines": []
    })) for sample in samples]
    records += [({"id": key}, result) for key, result in result_by_id.items() if key not in sample_by_id]
    report = {"samples": len(records), "missing_images": [], "pages": {}}
    filenames = {"ocr_pdf": "ocr.pdf", "recovered_pdf": "recovered.pdf",
                 "comparison_pdf": "comparison.pdf", "review_html": "review.html"}
    with TemporaryDirectory(prefix=".render-", dir=output_dir) as staging:
        staging = Path(staging)
        layouts = {}
        for field, filename, label in (("ocr_text", "ocr.pdf", "Visual OCR"),
                                       ("text", "recovered.pdf", "Recovered")):
            pdf = canvas.Canvas(str(staging / filename), pagesize=(width, height), pageCompression=1)
            pdf.setTitle(label)
            count = 0
            for sample, result in records:
                pages = _layout(result, field, width - 2 * margin, rows_per_page, size)
                layouts[(sample["id"], field)] = pages
                for number, rows in enumerate(pages, 1):
                    _draw_text_page(pdf, rows, sample["id"], label, number, len(pages),
                                    (width, height), margin, size, leading)
                    pdf.showPage()
                    count += 1
            if not count:
                _draw_text_page(pdf, ["[No samples were supplied.]"], "Empty run", label, 1, 1,
                                (width, height), margin, size, leading)
                pdf.showPage()
                count = 1
            pdf.save()
            report["pages"][filename] = count

        gap, border, heading = 12, 16, 38
        comparison_size = (4 * width + 3 * gap + 2 * border, height + heading + 2 * border)
        pdf = canvas.Canvas(str(staging / "comparison.pdf"), pagesize=comparison_size, pageCompression=1)
        pdf.setTitle("Clear / Out / Blur / Final")
        count, missing = 0, set()
        for sample, result in records:
            pages = layouts[(sample["id"], "text")]
            for number, rows in enumerate(pages, 1):
                pdf.setFillColorRGB(0, 0, 0)
                pdf.setFont("Helvetica-Bold", 11)
                pdf.drawString(border, comparison_size[1] - 18,
                               _wrap(f'{_pdf_text(sample["id"])} | {number}/{len(pages)}', comparison_size[0] - 2 * border, 11)[0])
                for column, (label, source) in enumerate((("Clear (reference only)", references.get(sample["id"])),
                                                         ("Out", sample.get("out")), ("Blur", sample.get("blur")))):
                    x = border + column * (width + gap)
                    pdf.setFont("Helvetica-Bold", 11)
                    pdf.drawString(x, height + border + 9, label)
                    missing_label = "Clear" if column == 0 else label
                    if not _draw_image(pdf, source, missing_label, (x, border, width, height)):
                        missing.add((sample["id"], missing_label))
                _draw_text_page(pdf, rows, sample["id"], "Final", number, len(pages),
                                (width, height), margin, size, leading,
                                offset=border + 3 * (width + gap), top=border)
                pdf.showPage()
                count += 1
        if not count:
            pdf.setFont("Helvetica", 12)
            pdf.drawString(border, comparison_size[1] - 32, "[No samples were supplied.]")
            pdf.showPage()
            count = 1
        pdf.save()
        report["pages"]["comparison.pdf"] = count
        report["missing_images"] = [{"id": key, "kind": kind} for key, kind in sorted(missing)]
        (staging / "review.html").write_text(_review_html(records, output_dir), encoding="utf-8")
        for key, filename in filenames.items():
            os.replace(staging / filename, output_dir / filename)
            report[key] = str(output_dir / filename)
    return report
