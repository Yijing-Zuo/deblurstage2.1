"""Small, CPU-only CTC decoders. Scores use original (unrenormalized) probabilities."""
from pathlib import Path

import numpy as np


def allowed_text(text, alphabet_string):
    return isinstance(text, str) and all(char in alphabet_string for char in text)


def validate_evidence(probs, alphabet, blank_id=0):
    probs = np.asarray(probs, dtype=np.float64)
    alphabet = [str(char) for char in alphabet]
    if probs.ndim != 2 or not len(probs) or probs.shape[1] != len(alphabet):
        raise ValueError("CTC probabilities and alphabet dimensions do not match")
    if not 0 <= blank_id < len(alphabet) or alphabet[blank_id] != "":
        raise ValueError("CTC blank must have an empty-string alphabet entry")
    if len(set(alphabet)) != len(alphabet) or any(len(char) != 1 for i, char in enumerate(alphabet) if i != blank_id):
        raise ValueError("CTC alphabet must contain distinct single characters and one blank")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("CTC probabilities must be finite values in [0, 1]")
    if (probs.sum(axis=1) > 1.0001).any():
        raise ValueError("CTC rows must be probabilities, not logits")
    return probs, alphabet


def load_evidence(path):
    with np.load(Path(path), allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    result["blank_id"] = int(result.get("blank_id", 0))
    probs, alphabet = validate_evidence(result["probs"], result["alphabet"], result["blank_id"])
    result.update(probs=probs, alphabet=alphabet)
    if "excluded_mass" in result:
        excluded = result["excluded_mass"]
        if excluded.shape != (len(probs),) or not np.isfinite(excluded).all() or (excluded < -1e-5).any() or (excluded > 1.0001).any():
            raise ValueError("Invalid excluded-character probability mass")
        if not np.allclose(probs.sum(axis=1) + excluded, 1, atol=2e-4):
            raise ValueError("Retained and excluded CTC probability mass do not sum to one")
    return result


def greedy_decode(probs, alphabet, blank_id=0):
    probs, alphabet = validate_evidence(probs, alphabet, blank_id)
    previous, result = None, []
    for frame in probs:
        char_id = int(frame.argmax())
        if frame[char_id] == 0:
            # No legal path exists; do not invent a character for an all-zero row.
            return ""
        if char_id != blank_id and char_id != previous:
            result.append(alphabet[char_id])
        previous = char_id
    return "".join(result)


def prefix_beam_search(probs, alphabet, blank_id=0, beam_width=8, top_k=8, token_top_k=12):
    """Approximate transcript search; sum blank/repeated paths before pruning.

    token_top_k limits per-frame search only. Rescore returned texts with the exact
    forward algorithm below before comparing visual evidence.
    """
    probs, alphabet = validate_evidence(probs, alphabet, blank_id)
    if min(beam_width, top_k, token_top_k) < 1:
        raise ValueError("Beam and candidate budgets must be positive")
    beams = {(): (0.0, -np.inf)}
    with np.errstate(divide="ignore"):
        logs = np.log(probs)
    for frame in logs:
        char_ids = np.argsort(frame)[-min(token_top_k, len(frame)):]
        char_ids = set(int(i) for i in char_ids if np.isfinite(frame[i]))
        char_ids.add(blank_id)
        following = {}

        def add(prefix, blank=-np.inf, nonblank=-np.inf):
            old_blank, old_nonblank = following.get(prefix, (-np.inf, -np.inf))
            following[prefix] = (float(np.logaddexp(old_blank, blank)), float(np.logaddexp(old_nonblank, nonblank)))

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = np.logaddexp(p_blank, p_nonblank)
            add(prefix, blank=total + frame[blank_id])
            for char_id in char_ids - {blank_id}:
                if prefix and char_id == prefix[-1]:
                    add(prefix, nonblank=p_nonblank + frame[char_id])
                    add(prefix + (char_id,), nonblank=p_blank + frame[char_id])
                else:
                    add(prefix + (char_id,), nonblank=total + frame[char_id])
        beams = dict(sorted(following.items(), key=lambda item: np.logaddexp(*item[1]), reverse=True)[:beam_width])
    return [{"text": "".join(alphabet[i] for i in prefix), "score": float(np.logaddexp(*scores))}
            for prefix, scores in sorted(beams.items(), key=lambda item: np.logaddexp(*item[1]), reverse=True)[:top_k]
            if np.isfinite(np.logaddexp(*scores))]


def ctc_log_probabilities(probs, alphabet, texts, blank_id=0, batch_size=32):
    """Exact CTC forward sums for full candidate lines; impossible paths are -inf.

    Batching candidates avoids one Python time-step loop per proposed word. This
    never adds epsilon to genuine zero probabilities or renormalizes classes.
    """
    probs, alphabet = validate_evidence(probs, alphabet, blank_id)
    texts = list(texts)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    index = {char: i for i, char in enumerate(alphabet) if i != blank_id}
    with np.errstate(divide="ignore"):
        logs = np.log(probs)
    scores = np.full(len(texts), -np.inf)
    valid = [i for i, text in enumerate(texts) if isinstance(text, str)
             and all(char in index for char in text)
             and len(text) + sum(a == b for a, b in zip(text, text[1:])) <= len(probs)]
    for offset in range(0, len(valid), batch_size):
        rows = valid[offset:offset + batch_size]
        lengths = np.asarray([2 * len(texts[i]) + 1 for i in rows])
        labels = np.full((len(rows), int(lengths.max())), blank_id, dtype=int)
        for row, i in enumerate(rows):
            labels[row, 1:lengths[row]:2] = [index[char] for char in texts[i]]
        active = np.arange(labels.shape[1])[None, :] < lengths[:, None]
        skip = np.zeros_like(active)
        skip[:, 2:] = (labels[:, 2:] != blank_id) & (labels[:, 2:] != labels[:, :-2])
        state = np.full(labels.shape, -np.inf)
        state[:, 0] = 0
        for frame in logs:
            one = np.full_like(state, -np.inf)
            two = np.full_like(state, -np.inf)
            one[:, 1:] = state[:, :-1]
            two[:, 2:] = state[:, :-2]
            following = np.logaddexp(state, one)
            following = np.logaddexp(following, np.where(skip, two, -np.inf))
            state = np.where(active, following + frame[labels], -np.inf)
        last = state[np.arange(len(rows)), lengths - 1]
        previous = state[np.arange(len(rows)), np.maximum(lengths - 2, 0)]
        scores[rows] = np.where(lengths > 1, np.logaddexp(last, previous), last)
    return scores.tolist()


def ctc_log_probability(probs, alphabet, text, blank_id=0):
    return ctc_log_probabilities(probs, alphabet, [text], blank_id)[0]
