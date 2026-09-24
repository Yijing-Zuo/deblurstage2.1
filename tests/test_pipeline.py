"""CPU regression tests with deterministic fake OCR; no weights or GPU inference."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from ocr import WidthLimitError, restrict_probabilities
from pipeline import OCRRunner, main, observation_anomalies, run_decode
from storage import (ALPHABET, load_samples, normalize_typography, pixel_hash, read_json,
                     read_jsonl, run_lock, write_jsonl)


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.interrupt = False
        self.width_limit = None

    def detect(self, path):
        return [{"polygon": [[2, 3], [118, 3], [118, 17], [2, 17]], "score": .9}]

    def recognize(self, path, model):
        self.calls.append(model)
        if self.interrupt and model == "b":
            raise KeyboardInterrupt()
        with Image.open(path) as image:
            if self.width_limit and image.width > self.width_limit:
                raise WidthLimitError("Use segments")
        text = "Read words"
        alphabet = [""] + sorted(set(text))
        probs = np.zeros((2 * len(text) + 1, len(alphabet)))
        probs[::2, 0] = 1
        for i, char in enumerate(text):
            probs[2 * i + 1, alphabet.index(char)] = 1
        return {"text": text, "raw_text": text}, {
            "probs": probs, "alphabet": np.asarray(alphabet), "blank_id": np.asarray(0),
            "excluded_mass": np.zeros(len(probs))}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = Image.new("RGB", (120, 30), "white")
        draw = ImageDraw.Draw(self.image)
        for x in range(4, 116, 6):
            draw.rectangle((x, 5, x + 2, 13), fill="black")
        self.path = self.root / "out.png"
        self.image.save(self.path)
        self.sample = {"id": "4_004", "document_id": "4", "out": str(self.path),
                       "blur": str(self.root / "MISSING_BLUR.png")}
        self.config = {"detection": {}, "layout": {"max_residual_regions": 0}, "rescue": {},
                       "decode": {"mode": "visual"}}
        self.models = [{"name": name, "file_hashes": {"weight": name}} for name in ("det", "a", "b")]
        self.backend = FakeBackend()

    def runner(self):
        return OCRRunner(self.config, self.root, self.models, self.backend)

    def test_interruption_resumes_saved_observation_without_clear_or_blur(self):
        self.backend.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            self.runner().page(self.sample)
        self.assertEqual(self.backend.calls, ["a", "b"])
        self.backend.interrupt = False
        page = self.runner().page(self.sample)
        self.assertEqual(self.backend.calls, ["a", "b", "b"])
        self.assertEqual(page["status"], "done")
        self.runner().page(self.sample)
        self.assertEqual(len(self.backend.calls), 3)

    def test_corrupt_crop_is_recreated_before_recognition(self):
        page = self.runner().page(self.sample)
        crop = self.root / page["lines"][0]["observations"][0]["crop"]
        Image.new("RGB", (4, 4), "red").save(crop)
        repaired = self.runner().page(self.sample)
        self.assertEqual(repaired["status"], "done")
        with Image.open(crop) as image:
            self.assertEqual(pixel_hash(image), crop.stem)

    def test_model_change_reuses_other_model_pixels(self):
        self.runner().page(self.sample)
        self.models[1]["file_hashes"]["weight"] = "new-a"
        self.runner().page(self.sample)
        self.assertEqual(self.backend.calls, ["a", "b", "a"])

    def test_decode_cache_preserves_new_line_identity_and_checks_corruption(self):
        page = self.runner().page(self.sample)
        decoded = run_decode([page], self.config, self.root)
        changed = copy.deepcopy(page)
        changed["lines"][0].update(id="new-line", order=99, column=3)
        again = run_decode([changed], self.config, self.root)
        self.assertEqual(again[0]["lines"][0]["id"], "new-line")
        self.assertEqual(again[0]["lines"][0]["order"], 99)
        self.assertEqual(again[0]["lines"][0]["text"], decoded[0]["lines"][0]["text"])
        obs = page["lines"][0]["observations"][0]
        (self.root / obs["probabilities"]).write_bytes(b"damaged")
        damaged = run_decode([page], self.config, self.root)
        self.assertEqual(damaged[0]["status"], "error")
        self.assertIn("hash mismatch", damaged[0]["lines"][0]["error"])

    def test_optional_detection_failure_keeps_successful_lines(self):
        detect = self.backend.detect
        def fail_roi(path):
            with Image.open(path) as image:
                if image.size != self.image.size:
                    raise RuntimeError("ROI failure")
            return detect(path)
        self.backend.detect = fail_roi
        with patch("layout.residual_regions", return_value=[[0, 20, 120, 30]]):
            page = self.runner().page(self.sample)
        self.assertEqual(len(page["lines"]), 1)
        self.assertEqual(page["status"], "error")
        self.assertIn("ROI failure", page["detection_errors"][0]["error"])

    def test_width_limit_rescue_is_bounded_and_exception_type_survives_cache(self):
        self.backend.width_limit = 60
        line = {"id": "line", "box": [0, 4, 120, 15], "order": 0, "column": 0}
        with patch("layout.tight_box", return_value=line["box"]), patch(
                "layout.segment_boxes", return_value=[[0, 4, 45, 15], [35, 4, 80, 15], [75, 4, 120, 15]]):
            result = self.runner().read_line(self.image, line)
        self.assertTrue(all(error["type"] == "WidthLimitError" for error in result["errors"]))
        self.assertEqual(result["status"], "done")
        self.assertLessEqual(len(self.backend.calls), 10)

    def test_long_colored_noise_does_not_trigger_short_read(self):
        image = Image.new("RGB", (300, 20), "white")
        ImageDraw.Draw(image).rectangle((0, 5, 299, 10), fill="red")
        self.assertEqual(observation_anomalies(image, {"text": "a"}, {})["anomalies"], [])

    def test_manifest_relative_paths_and_numpy_serialization(self):
        manifest = self.root / "samples.jsonl"
        write_jsonl(manifest, [{"id": "4_004", "out": "out.png", "page": np.int64(4),
                                "clear": "missing", "blur": "missing"}])
        samples = load_samples(manifest)
        self.assertEqual(samples[0]["out"], str(self.path))
        self.assertNotIn("clear", samples[0])
        self.assertEqual(read_jsonl(manifest)[0]["page"], 4)

    def test_writer_lock_releases_after_interruption(self):
        with self.assertRaises(KeyboardInterrupt):
            with run_lock(self.root):
                with self.assertRaises(RuntimeError):
                    with run_lock(self.root):
                        pass
                raise KeyboardInterrupt()
        with run_lock(self.root):
            pass

    def test_restriction_preserves_excluded_mass_and_rejects_logits(self):
        probs = np.asarray([[.1, .2, .7]])
        selected = restrict_probabilities(probs, ["blank", "a", "中"], ALPHABET)
        np.testing.assert_allclose(selected["probs"], [[.1, .2]])
        np.testing.assert_allclose(selected["excluded_mass"], [.7])
        with self.assertRaisesRegex(ValueError, "normalized"):
            restrict_probabilities(probs * 10, ["blank", "a", "中"], ALPHABET)

    def test_typography_normalization_does_not_transliterate_unknown_scripts(self):
        self.assertEqual(normalize_typography("\u201c\ufb01le\u201d\u2014中"), '"file"-中')

    def test_cli_all_renders_and_resumes_without_references(self):
        manifest, run = self.root / "samples.jsonl", self.root / "run"
        write_jsonl(manifest, [self.sample])
        config = {**self.config, "samples": str(manifest), "run_dir": str(run),
                  "model_cache": str(self.root / "models"), "models": self.models,
                  "device": "cpu", "references": str(self.root / "missing-clear.jsonl"),
                  "expected_samples": 1}
        with patch("pipeline.read_config", return_value=config), patch(
                "pipeline.resolve_models", return_value=self.models), patch(
                "pipeline.PaddleBackend", return_value=self.backend):
            self.assertEqual(main(["all", "--offline"]), 0)
            self.assertEqual(main(["all", "--offline"]), 0)
        self.assertEqual(self.backend.calls, ["a", "b"])
        self.assertEqual(read_json(run / "run.json")["completed_pages"], 1)
        for filename in ("ocr.pdf", "recovered.pdf", "comparison.pdf", "review.html", "lines.jsonl"):
            self.assertTrue((run / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
