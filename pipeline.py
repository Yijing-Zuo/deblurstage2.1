"""One entry point for a fresh Out-only experiment: download, ocr, decode, render, all."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from PIL import Image
import yaml

import layout
from ocr import ADAPTER_VERSION, PaddleBackend, WidthLimitError, resolve_models
from storage import (ALPHABET, SCHEMA, atomic_write, code_stamp, file_hash, fingerprint,
                     load_references, load_samples, pixel_hash, read_json, read_jsonl,
                     resolve_path, run_lock, validate_observation, write_json, write_jsonl, write_npz)

OCR_FLOW_VERSION = 1


def read_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    for field in ("samples", "model_cache", "run_dir", "references"):
        if config.get(field):
            config[field] = str(resolve_path(config[field], path.parent))
    if len(config["models"]) != 3 or len({m["name"] for m in config["models"]}) != 3:
        raise ValueError("Configure one detector and two distinct recognizers")
    return config


def alnum_count(text):
    return sum(c.isascii() and c.isalnum() for c in text)


def observation_anomalies(crop, metadata, settings):
    stats = layout.ink_stats(crop)
    count = alnum_count(metadata["text"])
    reasons = []
    if not metadata["text"].strip() and stats["text_like"]:
        reasons.append("empty_read")
    if (stats["text_like"] and count <= settings.get("short_chars", 5)
            and stats["extent_ratio"] >= settings.get("long_extent_ratio", 12)):
        reasons.append("short_read")
    if metadata.get("mean_excluded_mass", 0) > 0.4:
        reasons.append("excluded_characters")
    return {"anomalies": reasons, "ink_extent_ratio": stats["extent_ratio"]}


def needs_rescue(observations, settings, expected_models=2):
    if len(observations) < expected_models:
        return True
    if any(o.get("anomalies") for o in observations):
        return True
    lengths = [alnum_count(o["text"]) for o in observations]
    return (max(lengths, default=0) > settings.get("short_chars", 5)
            and min(lengths) / max(lengths) < settings.get("length_ratio", 0.35))


class OCRRunner:
    def __init__(self, config, run_dir, models, backend):
        self.config, self.run, self.models, self.backend = config, Path(run_dir), models, backend
        self.failed, self.validated = {}, set()
        self.adapter = {"version": ADAPTER_VERSION, **code_stamp("ocr.py", "storage.py")}
        self.model_keys = {m["name"]: m["file_hashes"] for m in models}

    def save_crop(self, crop):
        key = pixel_hash(crop)
        path = self.run / "crops" / f"{key}.png"
        valid = False
        if path.is_file():
            try:
                with Image.open(path) as saved:
                    valid = pixel_hash(saved) == key
            except OSError:
                pass
        if not valid:
            atomic_write(path, lambda stream: crop.save(stream, format="PNG"), binary=True)
        return path

    def detect(self, image):
        key = fingerprint({"pixels": pixel_hash(image), "detector": self.models[0]["file_hashes"],
                           "settings": self.config["detection"], "adapter": self.adapter})
        path = self.run / "cache" / "detection" / f"{key}.json"
        if path.is_file():
            saved = read_json(path)
            if saved.get("key") == key:
                return saved["detections"]
        detections = self.backend.detect(self.save_crop(image))
        write_json(path, {"key": key, "detections": detections})
        return detections

    def valid_observation(self, row):
        key = row["key"]
        if key in self.validated:
            return True
        try:
            validate_observation(row, self.run)
        except (OSError, ValueError, KeyError):
            return False
        self.validated.add(key)
        return True

    def observe(self, image, box, variant, model, segment_index=None):
        crop = image.crop(box).convert("RGB")
        key = fingerprint({"pixels": pixel_hash(crop), "model": self.model_keys[model],
                           "alphabet": ALPHABET, "adapter": self.adapter})
        meta_path = self.run / "cache" / "recognition" / f"{key}.json"
        saved = read_json(meta_path) if meta_path.is_file() else None
        if saved is None or saved.get("key") != key or not self.valid_observation(saved):
            if key in self.failed:
                raise self.failed[key]
            crop_path = self.save_crop(crop)
            try:
                metadata, values = self.backend.recognize(crop_path, model)
                from ctc import validate_evidence
                validate_evidence(values["probs"], values["alphabet"], int(values["blank_id"]))
            except Exception as exc:
                self.failed[key] = exc
                raise
            path = self.run / "cache" / "recognition" / f"{key}.npz"
            write_npz(path, values)
            saved = {**metadata, "key": key, "model": model,
                     "crop": crop_path.relative_to(self.run).as_posix(),
                     "crop_sha256": file_hash(crop_path),
                     "probabilities": path.relative_to(self.run).as_posix(), "npz_sha256": file_hash(path)}
            write_json(meta_path, saved)
            self.validated.add(key)
        result = {**saved, "variant": variant, "box": list(box),
                  **observation_anomalies(crop, saved, self.config["rescue"])}
        if segment_index is not None:
            result["segment_index"] = segment_index
        return result

    def read_line(self, image, line):
        observations, errors = [], []
        settings = self.config["rescue"]

        def read(box, variant, segment_index=None):
            current = []
            for model in self.models[1:]:
                try:
                    row = self.observe(image, box, variant, model["name"], segment_index)
                    current.append(row)
                    observations.append(row)
                except Exception as exc:
                    errors.append({"model": model["name"], "variant": variant, "box": list(box),
                                   "type": type(exc).__name__, "error": str(exc)})
            return current

        box = layout.crop_box(line["box"], image.size, padding=1)
        first = read(box, "full")
        rescued = needs_rescue(first, settings)
        if rescued:
            tight = layout.tight_box(image, box, padding=settings.get("tight_padding", 1))
            second = read(tight, "tight")
            if needs_rescue(second, settings):
                for i, segment in enumerate(layout.segment_boxes(
                        image, tight, max_segments=settings.get("max_segments", 3),
                        overlap=settings.get("overlap", 0.15))):
                    read(segment, "segment", i)
        missing = [m["name"] for m in self.models[1:]
                   if not any(o["model"] == m["name"] for o in observations)]
        # A width refusal repaired by short crops is a completed rescue, not a fatal error.
        failures = [e for e in errors if e["type"] != "WidthLimitError"]
        review = []
        if missing or failures:
            review.append("ocr_failure")
        if not observations:
            review.append("no_reading")
        if line.get("geometry_warning"):
            review.append(line["geometry_warning"])
        return {**line, "observations": observations, "errors": errors, "rescued": rescued,
                "status": "error" if missing or failures else "done", "needs_review": review}

    def page(self, sample):
        with Image.open(sample["out"]) as source:
            image = source.convert("RGB")
        identity = {"out_sha256": file_hash(sample["out"]), "pixels": pixel_hash(image),
                    "models": self.model_keys, "adapter": self.adapter,
                    "layout_code": code_stamp("layout.py"), "flow_version": OCR_FLOW_VERSION,
                    "layout": self.config["layout"], "detection": self.config["detection"],
                    "rescue": self.config["rescue"]}
        key = fingerprint(identity)
        path = self.run / "cache" / "pages" / f"{sample['id']}.json"
        if path.is_file():
            saved = read_json(path)
            if (saved.get("ocr_key") == key and saved.get("status") == "done"
                    and all(self.valid_observation(o) for line in saved["lines"] for o in line["observations"])):
                print(f"{sample['id']}: cached ({len(saved['lines'])} lines)", flush=True)
                return saved
        detections = self.detect(image)
        lines, suppressed = layout.build_lines(image, detections, self.config["layout"])
        regions = layout.residual_regions(image, lines, self.config["layout"])
        detection_errors = []
        for region in regions:
            try:
                extra = self.detect(image.crop(region))
            except Exception as exc:
                detection_errors.append({"box": region, "error": f"{type(exc).__name__}: {exc}"})
                continue
            for detection in extra:
                points = np.asarray(detection["polygon"], dtype=float) + np.asarray(region[:2])
                detections.append({**detection, "polygon": points.tolist(), "origin": "residual"})
        if regions:
            lines, suppressed = layout.build_lines(image, detections, self.config["layout"])
        results = [self.read_line(image, {**line, "id": f"{sample['id']}:l{i:04d}"})
                   for i, line in enumerate(lines)]
        coverage = layout.coverage(image, lines)
        review = []
        if coverage.get("uncovered_ink_bands"):
            review.append("uncovered_ink")
        if not results:
            review.append("no_lines_detected")
        if detection_errors:
            review.append("residual_detection_failure")
        saved = {"schema": SCHEMA, "id": sample["id"], "ocr_key": key,
                 "out_sha256": identity["out_sha256"], "size": list(image.size),
                 "status": "error" if not results or detection_errors or any(l["status"] == "error" for l in results) else "done",
                 "needs_review": review, "coverage": coverage, "residual_regions": regions,
                 "suppressed": suppressed, "detection_errors": detection_errors, "lines": results}
        write_json(path, saved)
        print(f"{sample['id']}: {saved['status']} ({len(results)} lines)", flush=True)
        return saved


def run_ocr(samples, config, run_dir, models, backend):
    runner, pages = OCRRunner(config, run_dir, models, backend), []
    for sample in samples:
        try:
            page = runner.page(sample)
        except Exception as exc:
            page = {"schema": SCHEMA, "id": sample["id"], "status": "error", "lines": [],
                    "out_sha256": file_hash(sample["out"]), "needs_review": ["page_failure"],
                    "error": f"{type(exc).__name__}: {exc}"}
            print(f"{sample['id']}: error ({page['error']})", flush=True)
        pages.append(page)
        write_jsonl(Path(run_dir) / "ocr.jsonl", pages)
    return pages


def check_pages(samples, pages):
    index = {page["id"]: page for page in pages}
    if len(index) != len(pages) or set(index) != {s["id"] for s in samples}:
        raise ValueError("Run pages do not match manifest; resume OCR to complete it")
    for sample in samples:
        if index[sample["id"]].get("out_sha256") != file_hash(sample["out"]):
            raise ValueError(f"{sample['id']}: Out changed; run OCR again")


def run_decode(pages, config, run_dir):
    from decode import decode_line, make_lexicon
    settings = config["decode"]
    started = time.monotonic()
    total = sum(len(page["lines"]) for page in pages)
    completed = 0
    print(f"[decode] {len(pages)} pages, {total} lines; CPU candidate scoring. "
          "Preparing dictionary...", flush=True)
    lexicon = make_lexicon(settings) if settings.get("mode") != "visual" else None
    print(f"[decode] ready ({time.monotonic() - started:.1f}s); saved lines will be reused.", flush=True)
    stamp = code_stamp("decode.py", "ctc.py")
    result, validated = [], set()
    fields = ("ocr_text", "text", "needs_review", "changes", "candidates", "visual_sources")
    for page_number, page in enumerate(pages, 1):
        current = copy.deepcopy(page)
        for line in current["lines"]:
            source = {"observations": line["observations"], "box": line["box"]}
            key = fingerprint({**source, "settings": settings, "code": stamp})
            cache = Path(run_dir) / "cache" / "decoded" / f"{key}.json"
            label = f"[decode {completed + 1}/{total}] {line['id']}"
            line_started = time.monotonic()
            print(f"{label}: {'checking saved result' if cache.is_file() else 'scoring candidates'}...", flush=True)
            outcome = "done"
            try:
                for observation in line["observations"]:
                    identity = (observation["key"], observation["npz_sha256"], observation["crop_sha256"])
                    if identity not in validated:
                        validate_observation(observation, run_dir)
                        validated.add(identity)
                if cache.is_file():
                    decoded = read_json(cache)
                    outcome = "cached"
                else:
                    decoded = decode_line(source, Path(run_dir), settings, lexicon)
                    decoded = {field: decoded[field] for field in fields if field in decoded}
                    write_json(cache, decoded)
                previous = line.get("needs_review", [])
                line.update({field: decoded[field] for field in fields if field in decoded})
                line["needs_review"] = sorted(set(previous + line.get("needs_review", [])))
            except Exception as exc:
                outcome = f"error ({type(exc).__name__}: {exc})"
                available = [o["text"] for o in line["observations"] if o.get("text")]
                fallback = available[0] if available else "[unreadable]"
                line.update(ocr_text=fallback, text=fallback, status="error",
                            needs_review=["decode_failure"], error=f"{type(exc).__name__}: {exc}", changes=[])
                current["status"] = "error"
            completed += 1
            print(f"{label}: {outcome} ({time.monotonic() - line_started:.1f}s)", flush=True)
        current["decode_stamp"] = stamp
        current["decode_settings"] = settings
        result.append(current)
        write_jsonl(Path(run_dir) / "pages.jsonl", result)
        print(f"[decode page {page_number}/{len(pages)}] {page['id']}: saved; "
              f"{completed}/{total} lines visited, {time.monotonic() - started:.1f}s elapsed.", flush=True)
    print("[decode] writing combined line results...", flush=True)
    write_jsonl(Path(run_dir) / "lines.jsonl",
                ({"sample_id": page["id"], **line} for page in result for line in page["lines"]))
    print(f"[decode] finished in {time.monotonic() - started:.1f}s.", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "ocr", "decode", "render", "all"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--samples")
    parser.add_argument("--run-dir")
    parser.add_argument("--documents", nargs="+")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--mode", choices=("visual", "conservative"))
    parser.add_argument("--no-references", action="store_true")
    args = parser.parse_args(argv)
    if args.mode and args.stage not in ("decode", "all"):
        parser.error("--mode applies only to decode/all; render uses saved decoded text")
    config = read_config(args.config)
    for key in ("samples", "run_dir"):
        if getattr(args, key):
            config[key] = str(Path(getattr(args, key)).expanduser().resolve())
    if args.mode:
        config["decode"]["mode"] = args.mode
    if args.stage == "download":
        models = resolve_models(config["models"], config["model_cache"], args.offline)
        print(f"Downloaded/verified {len(models)} pinned OCR models in {config['model_cache']}")
        return 0
    samples = load_samples(config["samples"], args.documents)
    if not args.documents and config.get("expected_samples") and len(samples) != config["expected_samples"]:
        raise ValueError(f"Expected {config['expected_samples']} pages, found {len(samples)}")
    run = Path(config["run_dir"])
    # Refuse to silently treat an old run as a fresh 2.1 cache.
    if (run.exists() and any(p.name not in {".writer.lock", "run.json.tmp"} for p in run.iterdir())
            and not (run / "run.json").is_file()):
        raise ValueError("Run directory is not initialized by 2.1; choose an empty directory")
    started = time.monotonic()
    with run_lock(run):
        manifest_key = fingerprint([{k: s[k] for k in ("id", "document_id")} for s in samples])
        state = read_json(run / "run.json") if (run / "run.json").is_file() else {
            "schema": SCHEMA, "manifest_key": manifest_key, "stages": {},
            "created_at": datetime.now(timezone.utc).isoformat()}
        if state.get("schema") != SCHEMA or state.get("manifest_key") != manifest_key:
            raise ValueError("Run belongs to another schema/manifest; choose a new run directory")
        state.update(config=config, sample_count=len(samples), last_stage=args.stage, status="running")
        state.pop("error", None)
        write_json(run / "run.json", state)
        write_json(run / "config.snapshot.json", config)
        try:
            if args.stage in ("ocr", "all"):
                print(f"[ocr] preparing models for {len(samples)} pages...", flush=True)
                models = resolve_models(config["models"], config["model_cache"], args.offline)
                backend = PaddleBackend(models, ALPHABET, config["device"], config["detection"])
                pages = run_ocr(samples, config, run, models, backend)
                state["models"] = models
            elif args.stage == "decode":
                print("[decode] reading saved OCR evidence...", flush=True)
                pages = read_jsonl(run / "ocr.jsonl")
            else:
                pages = read_jsonl(run / "pages.jsonl")
            print(f"[check] verifying {len(pages)} saved pages against Out inputs...", flush=True)
            check_pages(samples, pages)
            if args.stage in ("decode", "all"):
                pages = run_decode(pages, config, run)
            if args.stage in ("render", "all"):
                print("[render] creating OCR, recovered, comparison PDFs and review HTML...", flush=True)
                from render import render_run
                reference_path = config.get("references")
                references = {}
                if not args.no_references and reference_path:
                    if Path(reference_path).is_file():
                        references = load_references(reference_path)
                    else:
                        print("Clear manifest unavailable; comparison will mark Clear as missing.", flush=True)
                state["outputs"] = render_run(samples, pages, run, references, config.get("render"))
                print("[render] files saved.", flush=True)
            errors = sum(p["status"] == "error" for p in pages)
            review = sum(bool(p.get("needs_review")) or any(l.get("needs_review") for l in p["lines"]) for p in pages)
            state.update(status="incomplete" if errors else "done", completed_pages=len(pages) - errors,
                         error_pages=errors, review_pages=review)
            print(f"{args.stage}: {len(pages)} pages, {errors} errors, {review} needing review; {run}")
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                         error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            state["stages"][args.stage] = {"seconds": round(time.monotonic() - started, 2),
                                           "status": state["status"]}
            write_json(run / "run.json", state)
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
