"""Pinned PaddleX 3.7.0 CTC adapter; lazy GPU imports, isolated fresh model cache."""
from importlib.metadata import version
import math
import os
from pathlib import Path

import numpy as np

from storage import SCHEMA, file_hash, fingerprint, normalize_typography, read_json, write_json

MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
ADAPTER_VERSION = 1


def resolve_models(specs, cache, offline=False):
    from huggingface_hub import snapshot_download

    cache = Path(cache).resolve()
    marker = cache / "deblur21.json"
    if cache.exists() and any(cache.iterdir()) and not marker.is_file():
        raise ValueError("Model cache is not a deblur21 cache; choose a new empty directory")
    if marker.is_file() and read_json(marker).get("schema") != SCHEMA:
        raise ValueError("Incompatible model-cache schema")
    write_json(marker, {"schema": SCHEMA})
    # Explicit HF cache_dir prevents reuse of the old/global weights cache.
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache / "paddlex")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    result = []
    for spec in specs:
        revision = spec["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Model revision must be a full immutable commit SHA")
        directory = Path(snapshot_download(
            repo_id=spec["repo"], revision=revision, allow_patterns=list(MODEL_FILES),
            cache_dir=str(cache / "hub"), local_files_only=offline))
        hashes = {name: file_hash(directory / name) for name in MODEL_FILES}
        result.append({**spec, "path": str(directory), "file_hashes": hashes})
    write_json(cache / "models.json", {"schema": SCHEMA, "models": result})
    return result


def restrict_probabilities(prediction, characters, allowed):
    """Retain original softmax mass; excluded classes never inflate English confidence."""
    probs = np.asarray(prediction, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != len(characters) or characters[0] != "blank":
        raise ValueError("Unexpected CTC dimensions or blank symbol")
    if (not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1.0001
            or not np.allclose(probs.sum(axis=1), 1, atol=2e-4, rtol=0)):
        raise ValueError("Expected normalized CTC probabilities, not logits/NaNs")
    selected = [0] + [i for i, char in enumerate(characters) if i and len(char) == 1 and char in allowed]
    alphabet = [""] + [characters[i] for i in selected[1:]]
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("Duplicate character labels need an explicit mapping")
    omitted = np.ones(len(characters), dtype=bool)
    omitted[selected] = False
    return {"probs": probs[:, selected].copy(), "alphabet": np.asarray(alphabet),
            "blank_id": np.asarray(0, dtype=np.int64),
            "excluded_mass": probs[:, omitted].sum(axis=1, dtype=np.float32)}


def _greedy(probabilities, characters):
    previous, result = -1, []
    for index in np.asarray(probabilities).argmax(axis=-1):
        if index and index != previous:
            result.append(characters[index])
        previous = index
    return "".join(result)


class WidthLimitError(ValueError):
    """The pipeline must split this crop, not silently squeeze its characters."""


class PaddleBackend:
    def __init__(self, models, alphabet, device="gpu:0", detection=None):
        if version("paddlex") != "3.7.0":
            raise RuntimeError("The CTC adapter requires paddlex==3.7.0")
        from paddlex import create_predictor

        self.models, self.alphabet = models, alphabet
        self.detector = create_predictor(
            models[0]["name"], model_dir=models[0]["path"], device=device,
            engine="paddle_static", batch_size=1, **(detection or {}))
        self.recognizers = {}
        for spec in models[1:]:
            predictor = create_predictor(
                spec["name"], model_dir=spec["path"], device=device,
                engine="paddle_static", batch_size=1)
            if not all(hasattr(predictor, attr) for attr in ("runner", "pre_tfs", "post_op")):
                raise RuntimeError("Unexpected PaddleX CTC interface")
            self.recognizers[spec["name"]] = predictor

    def detect(self, path):
        results = list(self.detector.predict(str(path)))
        if len(results) != 1:
            raise RuntimeError("Expected exactly one detection result")
        result = results[0]
        if len(result["dt_polys"]) != len(result["dt_scores"]):
            raise ValueError("Detection polygons and scores differ in length")
        return [{"polygon": np.asarray(p).tolist(), "score": float(s)}
                for p, s in zip(result["dt_polys"], result["dt_scores"])]

    def recognize(self, path, model):
        predictor = self.recognizers[model]
        characters = list(predictor.post_op.character)
        # The model's Read operator handles BGR/RGB; never pass ambiguous RGB arrays.
        image = predictor.pre_tfs["Read"](imgs=[str(path)])[0]
        resize = predictor.pre_tfs["ReisizeNorm"]
        normalized = resize(imgs=[image])
        shape = normalized[0].shape
        natural_width = math.ceil(shape[1] * image.shape[1] / image.shape[0])
        if natural_width > resize.max_imgW:
            raise WidthLimitError(f"Natural width {natural_width} exceeds {resize.max_imgW}; split crop")
        inputs = predictor.pre_tfs["ToBatch"](imgs=normalized)
        probs = np.asarray(predictor.runner(x=inputs)[0])
        if probs.ndim != 3 or probs.shape[0] != 1:
            raise RuntimeError("Expected [1,T,C] CTC output")
        values = restrict_probabilities(probs[0], characters, self.alphabet)
        metadata = {
            "text": _greedy(values["probs"], values["alphabet"]),
            "raw_text": _greedy(probs[0], characters),
            "alphabet_hash": fingerprint(characters), "crop_shape": list(image.shape),
            "resized_height": int(shape[1]), "canvas_width": int(shape[2]),
            "resized_content_width": min(int(shape[2]), natural_width),
            "width_compressed": False, "time_steps": int(probs.shape[1]),
            "mean_excluded_mass": float(values["excluded_mass"].mean()),
        }
        metadata["normalized_raw_text"] = normalize_typography(metadata["raw_text"])
        return metadata, values
