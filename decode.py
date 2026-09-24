"""Bounded OCR alternatives, with a weak dictionary prior behind visual vetoes.

No Clear, Blur, downloaded language model or previous-run predictions are read.
All CTC comparisons are relative to the current text within the same observation;
overlapping observations are grouped by model, never counted as extra votes.
"""
import hashlib
import math
import re
import string
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from ctc import ctc_log_probabilities, load_evidence, prefix_beam_search

ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "
CONFUSIONS = (("rn", "m"), ("cl", "d"), ("vv", "w"), ("1", "l"), ("1", "I"), ("0", "O"))
DEFAULTS = dict(mode="conservative", beam_width=8, beam_top_k=3, max_candidates=12,
                phrase_words=3, phrase_beam=16, lexicon_max_edit=2,
                visual_tolerance=0.35, model_veto=0.7, prior_weight=0.06,
                min_gain=0.01, min_overlap_letters=4, min_model_support=2)


class Lexicon:
    """Generic bundled word/bigram counts; loaded only for conservative decoding."""
    def __init__(self, settings):
        from symspellpy import SymSpell

        self.engine = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        root = files("symspellpy")
        unigram = settings.get("lexicon_path") or root.joinpath("frequency_dictionary_en_82_765.txt")
        bigram = settings.get("bigram_path") or root.joinpath("frequency_bigramdictionary_en_243_342.txt")
        if not self.engine.load_dictionary(str(unigram), 0, 1):
            raise ValueError(f"Cannot load English word counts: {unigram}")
        if not self.engine.load_bigram_dictionary(str(bigram), 0, 2):
            raise ValueError(f"Cannot load English bigram counts: {bigram}")
        self.words, self.bigrams = self.engine.words, self.engine.bigrams

    @lru_cache(maxsize=8192)
    def suggest(self, word, max_edit=2, limit=6):
        from symspellpy import Verbosity

        found = self.engine.lookup(word.lower(), Verbosity.ALL, max_edit_distance=max_edit)
        return [item.term for item in found if item.term != word.lower()][:limit]

    def prior(self, text):
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
        if not words:
            return 0.0
        unigram = sum(math.log1p(self.words.get(word, 0)) for word in words) / len(words)
        pairs = list(zip(words, words[1:]))
        bigram = sum(math.log1p(self.bigrams.get(a + " " + b, 0)) for a, b in pairs) / max(1, len(pairs))
        return (unigram + 0.5 * bigram) / 25.0


def make_lexicon(settings):
    return None if settings.get("mode", "conservative") == "visual" else Lexicon(settings)


def _path(root, value):
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Evidence paths must be relative to this run")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence path escapes this run")
    return path


def _boundaries(before, after):
    result = {}
    for tag, start, end, other, stop in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        result[start], result[end] = other, stop
        if tag == "equal":
            result.update((start + i, other + i) for i in range(end - start + 1))
    if before:
        # Inserts at a complete line's edges belong to its first/last span.
        # Internal boundaries retain SequenceMatcher's mapping, avoiding overlap.
        result[0], result[len(before)] = 0, len(after)
    return result


def stitch_segments(observations, min_letters=4):
    """Return text plus exact source spans, or None; never concatenate uncertain seams."""
    parts = sorted(observations, key=lambda item: item.get("segment_index", -1))
    if not 2 <= len(parts) <= 3 or [p.get("segment_index") for p in parts] != list(range(len(parts))):
        return None
    text = parts[0]["text"]
    spans = [(parts[0], 0, len(text))]
    for previous, part in zip(parts, parts[1:]):
        a, b = previous["box"], part["box"]
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        if overlap <= 0 or b[0] <= a[0] or b[2] <= a[2]:
            return None
        right = part["text"]
        matches = [n for n in range(1, min(len(previous["text"]), len(right)) + 1)
                   if text[-n:] == right[:n] and sum(c.isalnum() for c in right[:n]) >= min_letters]
        if not matches:
            return None
        count = max(matches)
        # A tiny geometric overlap cannot substantiate a repeated half sentence.
        expected = len(right) * overlap / max(1, b[2] - b[0])
        if count > max(12, 2.5 * expected):
            return None
        start = len(text) - count
        text += right[count:]
        spans.append((part, start, len(text)))
    return {"text": text, "parts": spans, "variant": "segments"}


def _view_quality(view):
    observations = [part[0] for part in view["parts"]]
    severe = sum(any(flag in item.get("anomalies", []) for flag in ("empty_read", "short_read"))
                 for item in observations)
    excluded = max(float(np.mean(item["evidence"].get("excluded_mass", [0]))) for item in observations)
    # This orders views within a model only, and does not favor longer gibberish.
    return (severe, excluded > 0.3, view["variant"] != "full", view["variant"] == "segments")


def _scores(view, texts):
    totals, valid = np.zeros(len(texts)), np.ones(len(texts), dtype=bool)
    for item, start, end in view["parts"]:
        slices = []
        for i, text in enumerate(texts):
            if view["variant"] != "segments":
                slices.append(text)
                continue
            positions = _boundaries(view["text"], text)
            if start not in positions or end not in positions:
                valid[i] = False
                slices.append("")
            else:
                slices.append(text[positions[start]:positions[end]])
        ev = item["evidence"]
        scores = np.asarray(ctc_log_probabilities(ev["probs"], ev["alphabet"], slices, ev["blank_id"]))
        valid &= np.isfinite(scores)
        totals += np.where(np.isfinite(scores), scores, 0)
    # Overlapping segment scores provide one model-level comparison, not votes.
    totals /= len(view["parts"])
    return np.where(valid, totals, -np.inf)


def _alternatives(original, lexicon, config):
    if not any(c.isalpha() for c in original):
        return []  # Pure years, page numbers and codes are not spell-corrected.
    result = []
    for before, after in CONFUSIONS + tuple((b, a) for a, b in CONFUSIONS):
        for match in re.finditer(re.escape(before), original):
            result.append((original[:match.start()] + after + original[match.end():], "confusion"))
    match = re.fullmatch(r"([^\w]*)([A-Za-z0-9]+)([^\w]*)", original)
    if lexicon is None or not match:
        return result
    left, word, right = match.groups()
    if word.isupper() or any(c.isdigit() for c in word):
        return result  # Keep acronym and mixed identifier alternatives visual.
    distance = min(config["lexicon_max_edit"], 1 if len(word) < 5 else 3)
    for value in lexicon.suggest(word, max_edit=distance, limit=6):
        if word.istitle():
            value = value.capitalize()
        result.append((left + value + right, "lexicon"))
    for split in range(2, len(word) - 1):
        if word[:split].lower() in lexicon.words and word[split:].lower() in lexicon.words:
            result.append((left + word[:split] + " " + word[split:] + right, "split"))
    return result


def _pools(base, readings, lexicon, config):
    tokens = list(re.finditer(r"\S+", base))
    aligned = [(text, origin, _boundaries(base, text)) for text, origin in readings]
    pools = []
    for token in tokens:
        start, end, original = token.start(), token.end(), token.group()
        direct, beam = {}, {}
        for text, origin, boundaries in aligned:
            if start in boundaries and end in boundaries:
                value = text[boundaries[start]:boundaries[end]]
                if value.strip() and len(value) <= max(20, 2 * len(original)) and len(value.split()) <= 3:
                    (beam if origin.startswith("beam:") else direct).setdefault(value, origin)
        pool = {original: "original"}
        pool.update(direct)
        pool[original] = "original"
        sources = [list(beam.items()), _alternatives(original, lexicon, config)]
        # Direct OCR readings stay; spare slots are shared by beams and lexical alternatives.
        while any(sources) and len(pool) < config["max_candidates"]:
            for source in sources:
                if source and len(pool) < config["max_candidates"]:
                    value, origin = source.pop(0)
                    if value and all(c in ALPHABET for c in value):
                        pool.setdefault(value, origin)
        pools.append((start, end, list(pool.items())))
    return pools


def _relative(views, original, texts, changed_chars):
    deltas = {}
    for model, view in views.items():
        values = _scores(view, [original] + texts)
        if not np.isfinite(values[0]):
            deltas[model] = [None] * len(texts)
        else:
            deltas[model] = [float((value - values[0]) / max(1, count)) if np.isfinite(value) else None
                             for value, count in zip(values[1:], changed_chars)]
    return deltas


def _edit_count(before, after):
    return sum(max(end - start, stop - other) for tag, start, end, other, stop
               in SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if tag != "equal")


def decode_line(line, run_dir, settings, lexicon=None):
    """Decode one evidence record; malformed/missing evidence raises an explicit error."""
    config = DEFAULTS | settings
    if config["mode"] not in {"visual", "conservative"}:
        raise ValueError("decode.mode must be visual or conservative")
    if not 2 <= config["phrase_words"] <= 5 or min(config[k] for k in ("max_candidates", "phrase_beam", "beam_width", "beam_top_k")) < 1:
        raise ValueError("Invalid local decoding budgets")
    root = Path(run_dir).resolve()
    reasons = list(line.get("needs_review", []))
    grouped, readings = defaultdict(list), []
    for source in line.get("observations", []):
        if source.get("error"):
            raise ValueError(f"Failed OCR observation: {source['error']}")
        path = _path(root, source["probabilities"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["npz_sha256"]:
            raise ValueError(f"CTC evidence hash mismatch: {path.name}")
        if not _path(root, source["crop"]).is_file():
            raise ValueError(f"Missing OCR crop: {source['crop']}")
        observation = dict(source, evidence=load_evidence(path))
        if observation.get("variant") not in {"full", "tight", "segment"}:
            raise ValueError("Unknown OCR view variant")
        if any(c not in ALPHABET for c in observation["text"]):
            raise ValueError("OCR text contains characters outside the line alphabet")
        grouped[observation["model"]].append(observation)
    if not grouped:
        raise ValueError("Line has no OCR evidence")
    views = {}
    for model, observations in grouped.items():
        alternatives = [{"text": item["text"], "variant": item["variant"],
                         "parts": [(item, 0, len(item["text"]))]}
                        for item in observations if item["variant"] in {"full", "tight"}]
        segments = [item for item in observations if item["variant"] == "segment"]
        if segments:
            stitched = stitch_segments(segments, config["min_overlap_letters"])
            if stitched:
                box = line["box"]
                if segments[0]["box"][0] <= box[0] + 3 and max(s["box"][2] for s in segments) >= box[2] - 3:
                    alternatives.append(stitched)
                else:
                    reasons.append(f"segment_coverage:{model}")
            else:
                reasons.append(f"segment_alignment:{model}")
        usable = [item for item in alternatives if item["text"].strip()]
        if not usable:
            reasons.append(f"empty_model:{model}")
            continue
        views[model] = min(usable, key=_view_quality)
        for observation, _, _ in views[model]["parts"]:
            reasons.extend(f"{flag}:{model}" for flag in observation.get("anomalies", []))
        for view in alternatives:
            readings.append((view["text"], f"ocr:{model}:{view['variant']}"))
            if view["variant"] != "segments":
                normalized = view["parts"][0][0].get("normalized_raw_text", "")
                if normalized and normalized != view["text"] and all(c in ALPHABET for c in normalized):
                    # A textual candidate only; never relabel or renormalize the CTC tensor.
                    readings.append((normalized, f"ocr:{model}:typography"))
        # Beam alternatives are for complete views only. Segment evidence stays local.
        if views[model]["variant"] != "segments":
            ev = views[model]["parts"][0][0]["evidence"]
            for item in prefix_beam_search(ev["probs"], ev["alphabet"], ev["blank_id"],
                                          config["beam_width"], config["beam_top_k"]):
                if all(c in ALPHABET for c in item["text"]):
                    readings.append((item["text"], f"beam:{model}"))
    if not views:
        return dict(line, ocr_text="", text="", needs_review=list(dict.fromkeys(reasons + ["unreadable"])), changes=[], candidates=[])
    originals = list(dict.fromkeys(view["text"] for view in views.values()))
    provenance = {model: {"variant": view["variant"], "text": view["text"],
                          "crops": [item[0]["crop"] for item in view["parts"]]}
                  for model, view in views.items()}
    # Ordinal per-model ranks choose a visual anchor without comparing raw model scores.
    rank = np.zeros(len(originals))
    for view in views.values():
        values = _scores(view, originals)
        rank += np.asarray([sum(other > value for other in values) for value in values]) / max(1, len(originals) - 1)
    base = originals[int(np.argmin(rank))]
    if len(originals) > 1:
        reasons.append("model_disagreement")
    if config["mode"] == "visual":
        return dict(line, ocr_text=base, text=base, needs_review=list(dict.fromkeys(reasons)), changes=[],
                    visual_sources=provenance,
                    candidates=[{"text": text, "origin": origin} for text, origin in readings])
    if lexicon is None:
        lexicon = make_lexicon(config)
    pools = _pools(base, readings, lexicon, config)
    current, shift, changes, menus = base, 0, [], []
    for offset in range(0, len(pools), config["phrase_words"]):
        window = pools[offset:offset + config["phrase_words"]]
        start, end = window[0][0], window[-1][1]
        # Keep the original plus every single edit; bounded products add joint edits.
        phrase = base[start:end]
        pool = {phrase: "original"}
        # Preserve complete observed phrases before the lexical combination beam.
        # Keeping each observed word separately does not preserve their joint reading.
        for reading, origin in readings:
            if not origin.startswith("ocr:"):
                continue
            boundaries = _boundaries(base, reading)
            if start in boundaries and end in boundaries:
                value = reading[boundaries[start]:boundaries[end]]
                if value.strip() and len(value.split()) <= 5 and len(value) <= max(32, 2 * len(phrase)):
                    pool.setdefault(value, origin)
        partial = [("", 0.0)]
        for position, (a, b, options) in enumerate(window):
            for text, origin in options:
                pool.setdefault(base[start:a] + text + base[b:end], origin)
            gap = base[window[position - 1][1]:a] if position else ""
            expanded = [(prefix + gap + value, lexicon.prior(prefix + gap + value)) for prefix, _ in partial for value, _ in options]
            partial = sorted(expanded, key=lambda item: item[1], reverse=True)[:config["phrase_beam"]]
        for value, _ in partial:
            pool.setdefault(value, "phrase")
        # Adjacent word merging is a bounded local option, never applied globally.
        for first, second in zip(window, window[1:]):
            a, b, _ = first
            c, d, _ = second
            merged = base[a:b] + base[c:d]
            if merged.isalpha() and merged.lower() in lexicon.words:
                pool.setdefault(base[start:a] + merged + base[d:end], "merge")
        values = list(pool)
        location, stop = start + shift, end + shift
        candidates = [current[:location] + value + current[stop:] for value in values]
        deltas = _relative(views, current, candidates, [_edit_count(phrase, value) for value in values])
        records, best, best_gain = [], 0, 0.0
        old_prior = lexicon.prior(phrase)
        for i, value in enumerate(values):
            scores = {model: results[i] for model, results in deltas.items()}
            finite = [score for score in scores.values() if score is not None]
            complete = len(finite) == len(views)
            minimum = min(finite, default=-math.inf)
            visual = sum(finite) / len(finite) if finite else -math.inf
            prior = lexicon.prior(value) - old_prior
            gain = visual + config["prior_weight"] * max(-1, min(1, prior))
            accepted = (complete and len(finite) >= config["min_model_support"]
                        and minimum >= -config["model_veto"] and visual >= -config["visual_tolerance"])
            # A dictionary may break a visual near tie, not override an opposed model.
            if i and accepted and gain > max(best_gain + 1e-9, config["min_gain"]):
                best, best_gain = i, gain
            records.append({"text": value, "origin": pool[value], "relative_ctc": scores,
                            "prior_delta": prior, "eligible": bool(accepted)})
        if best:
            value = values[best]
            changes.append({"start": start, "end": end, "before": phrase, "after": value,
                            "relative_ctc": records[best]["relative_ctc"], "origin": pool[value]})
            current = candidates[best]
            shift += len(value) - len(phrase)
        elif any(item["text"] != phrase and any(score is not None and score >= -config["visual_tolerance"]
                                                for score in item["relative_ctc"].values()) for item in records):
            reasons.append("unresolved_candidates")
        menus.append({"start": start, "end": end, "original": phrase, "selected": values[best], "items": records})
    if any(word.lower() not in lexicon.words for word in re.findall(r"[A-Za-z]{3,}", current)):
        reasons.append("out_of_lexicon")  # A review hint; names are not forcibly replaced.
    return dict(line, ocr_text=base, text=current, needs_review=list(dict.fromkeys(reasons)),
                changes=changes, candidates=menus, visual_sources=provenance)
