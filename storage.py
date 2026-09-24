"""Small, explicit data paths and atomic files. No model or reference-image imports."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import string

import numpy as np

ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "
SCHEMA = "deblurstage2.1/1"
TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ",
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
})


def normalize_typography(text):
    # Explicit typography only: unknown scripts are not transliterated to English.
    return text.translate(TYPOGRAPHY)


def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not a JSON scalar: {type(value).__name__}")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=_scalar, sort_keys=True)


def fingerprint(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def code_stamp(*names):
    return {name: file_hash(Path(__file__).parent / name) for name in names}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        kwargs = {} if binary else {"encoding": "utf-8", "newline": "\n"}
        with temporary.open("wb" if binary else "w", **kwargs) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write(dumps(value) + "\n"))


def write_jsonl(path, rows):
    def write(stream):
        for row in rows:
            stream.write(dumps(row) + "\n")
    atomic_write(path, write)


def write_npz(path, values):
    atomic_write(path, lambda stream: np.savez_compressed(stream, **values), binary=True)


def validate_observation(row, directory):
    """Verify saved pixels and probabilities before using a derived result."""
    from ctc import load_evidence

    root = Path(directory).resolve()
    for field, checksum in (("crop", "crop_sha256"), ("probabilities", "npz_sha256")):
        relative = Path(row[field])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Evidence path escapes this run")
        if file_hash(path) != row[checksum]:
            raise ValueError(f"Evidence hash mismatch: {field}")
    load_evidence(root / row["probabilities"])


def resolve_path(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(parent) / path).resolve()


def load_samples(path, documents=None):
    """Only Out is required/validated here. Blur is resolved solely for rendering."""
    path = Path(path).resolve()
    result, seen = [], set()
    for row in read_jsonl(path):
        sample_id = str(row.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample_id) or sample_id in seen:
            raise ValueError(f"Invalid or duplicate sample id: {sample_id!r}")
        seen.add(sample_id)
        document = str(row.get("document_id", sample_id.split("_")[0]))
        if documents and document not in documents:
            continue
        if row.get("missing_tiles") or row.get("missing_mask"):
            raise ValueError(f"{sample_id}: incomplete tiles are outside this complete-page experiment")
        if not row.get("out"):
            raise ValueError(f"{sample_id}: missing Out path")
        out = resolve_path(row["out"], path.parent)
        if not out.is_file():
            raise FileNotFoundError(f"{sample_id}: Out image is missing: {out}")
        item = {"id": sample_id, "document_id": document, "out": str(out)}
        if row.get("blur"):
            item["blur"] = str(resolve_path(row["blur"], path.parent))
        result.append(item)
    if not result:
        raise ValueError("Sample manifest selected no complete Out pages")
    return result


def load_references(path):
    """Called by render only, never by OCR or decode."""
    if path is None:
        return {}
    path = Path(path).resolve()
    result = {}
    for row in read_jsonl(path):
        if row["id"] in result:
            raise ValueError(f"Duplicate Clear reference: {row['id']}")
        result[row["id"]] = str(resolve_path(row["clear"], path.parent))
    return result


@contextmanager
def run_lock(directory):
    """One writer; OS releases the lock even if Python is killed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another process is writing this run directory") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another process is writing this run directory") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
