import hashlib
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ctc import ctc_log_probability, greedy_decode, load_evidence, prefix_beam_search
from decode import decode_line, stitch_segments


class FakeLexicon:
    words = {"were": 1000000, "wore": 100, "cat": 10000, "the": 10000, "rain": 10000}

    def suggest(self, word, **kwargs):
        return {"wore": ["were"], "cxt": ["cat"], "therain": ["the rain"]}.get(word, [])

    def prior(self, text):
        return sum(word in {"were", "cat", "the", "rain"} for word in text.split()) / max(1, len(text.split()))


class CTCProbabilityTests(unittest.TestCase):
    def test_repeated_letters_need_separate_paths(self):
        alphabet = ["", "a"]
        probs = np.asarray([[.2, .8], [.7, .3], [.2, .8]])
        # Enumerate every path independently; 'aa' requires the intervening blank.
        total = 0
        for path in itertools.product(range(2), repeat=3):
            collapsed = "".join(alphabet[value] for i, value in enumerate(path)
                                if value and (not i or value != path[i - 1]))
            if collapsed == "aa":
                total += np.prod([probs[i, value] for i, value in enumerate(path)])
        self.assertAlmostEqual(np.exp(ctc_log_probability(probs, alphabet, "aa")), total)
        self.assertEqual(greedy_decode(probs, alphabet), "aa")
        beam = {item["text"]: item["score"] for item in prefix_beam_search(probs, alphabet, beam_width=8)}
        self.assertAlmostEqual(np.exp(beam["aa"]), total)
        self.assertGreater(beam["a"], beam["aa"])  # Several collapsed paths outweigh the greedy path.
        self.assertEqual(ctc_log_probability(probs[:2], alphabet, "aa"), -np.inf)

    def test_excluded_mass_is_not_renormalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ctc.npz"
            np.savez(path, probs=np.asarray([[.1, .3]]), alphabet=np.asarray(["", "a"]),
                     blank_id=0, excluded_mass=np.asarray([.6]))
            saved = load_evidence(path)
            self.assertAlmostEqual(ctc_log_probability(saved["probs"], saved["alphabet"], "a"), np.log(.3))
            np.savez(path, probs=np.asarray([[.1, .3]]), alphabet=np.asarray(["", "a"]),
                     blank_id=0, excluded_mass=np.asarray([.2]))
            with self.assertRaisesRegex(ValueError, "sum to one"):
                load_evidence(path)


class LocalDecodeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def observation(self, text, model="a", variant="full", box=None, segment_index=None,
                    alternatives=None, anomalies=None):
        alternatives = alternatives or {}
        alphabet = [""] + sorted(set(text) | set("".join(alternatives.values())))
        probs = np.zeros((len(text) * 2 + 1, len(alphabet)))
        probs[::2, 0] = 1
        for i, char in enumerate(text):
            alternate = alternatives.get(i)
            probs[2 * i + 1, alphabet.index(char)] = .51 if alternate else .99
            probs[2 * i + 1, 0] = .01
            if alternate:
                probs[2 * i + 1, alphabet.index(alternate)] = .48
        name = str(len(list(self.root.glob("*.npz"))))
        path = self.root / (name + ".npz")
        np.savez(path, probs=probs, alphabet=np.asarray(alphabet), blank_id=0,
                 excluded_mass=1 - probs.sum(axis=1))
        (self.root / (name + ".png")).write_bytes(b"crop fixture")
        result = dict(model=model, variant=variant, text=text, raw_text=text,
                      box=box or [0, 0, 100, 15], crop=name + ".png", probabilities=path.name,
                      npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), anomalies=anomalies or [])
        if segment_index is not None:
            result["segment_index"] = segment_index
        return result

    def line(self, *observations):
        return dict(id="sample:l0000", box=[0, 0, 100, 15], order=0, column=0,
                    observations=list(observations))

    def test_common_word_cannot_overrule_correct_visual_word(self):
        line = self.line(self.observation("wore", "a"), self.observation("wore", "b"))
        result = decode_line(line, self.root, {}, FakeLexicon())
        self.assertEqual(result["ocr_text"], "wore")
        self.assertEqual(result["text"], "wore")
        self.assertEqual(result["changes"], [])
        self.assertTrue(any(item["text"] == "were" and not item["eligible"]
                            for menu in result["candidates"] for item in menu["items"]))

    def test_weak_prior_only_resolves_visual_near_tie(self):
        observations = [self.observation("cxt", model, alternatives={1: "a"}) for model in ("a", "b")]
        result = decode_line(self.line(*observations), self.root,
                             {"prior_weight": .1}, FakeLexicon())
        self.assertEqual(result["ocr_text"], "cxt")
        self.assertEqual(result["text"], "cat")

    def test_other_model_vetoes_dictionary_change(self):
        line = self.line(self.observation("cxt", "a", alternatives={1: "a"}), self.observation("cxt", "b"))
        result = decode_line(line, self.root, {"prior_weight": 100}, FakeLexicon())
        self.assertEqual(result["text"], "cxt")

    def test_extra_views_of_one_model_are_not_extra_votes(self):
        repeated = [self.observation("cxt", "a", variant="tight", alternatives={1: "a"}) for _ in range(8)]
        line = self.line(self.observation("cxt", "a", alternatives={1: "a"}),
                         *repeated, self.observation("cxt", "b"))
        result = decode_line(line, self.root, {"prior_weight": 100}, FakeLexicon())
        self.assertEqual(result["text"], "cxt")
        self.assertEqual(len(result["visual_sources"]), 2)

    def test_joint_observed_phrase_survives_lexical_beam_pruning(self):
        class DistractingLexicon:
            words = dict.fromkeys(["aaa", "bbb", "ccc", "ddd", "eee", "fff"], 1000000)

            def suggest(self, word, **kwargs):
                return list(self.words)

            def prior(self, text):
                return sum(word in self.words for word in text.split()) / max(1, len(text.split()))

        first = self.observation("cat dog pig", "a", alternatives={0: "b", 4: "f", 8: "w"})
        second = self.observation("bat fog wig", "b", alternatives={0: "c", 4: "d", 8: "p"})
        path = self.root / second["probabilities"]
        with np.load(path) as saved:
            evidence = {key: saved[key].copy() for key in saved.files}
        alphabet = evidence["alphabet"].tolist()
        for position, own, other in ((0, "b", "c"), (4, "f", "d"), (8, "w", "p")):
            evidence["probs"][2 * position + 1, alphabet.index(own)] = .9
            evidence["probs"][2 * position + 1, alphabet.index(other)] = .09
        np.savez(path, **evidence)
        second["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = decode_line(self.line(first, second), self.root, {"phrase_beam": 2}, DistractingLexicon())
        self.assertEqual(result["ocr_text"], "cat dog pig")
        self.assertEqual(result["text"], "bat fog wig")
        selected = next(item for item in result["candidates"][0]["items"] if item["text"] == "bat fog wig")
        self.assertEqual(selected["origin"], "ocr:b:full")

    def test_normalized_typography_retains_both_edge_insertions(self):
        observations = [self.observation("cat", model) for model in ("a", "b")]
        for observation in observations:
            observation["normalized_raw_text"] = '"cat"'
        result = decode_line(self.line(*observations), self.root, {}, FakeLexicon())
        candidates = [item for menu in result["candidates"] for item in menu["items"]]
        normalized = next(item for item in candidates if item["text"] == '"cat"')
        self.assertIn("typography", normalized["origin"])
        self.assertFalse(normalized["eligible"])  # Candidate text cannot fabricate missing quote evidence.
        self.assertEqual(result["text"], "cat")

    def test_direct_reading_keeps_a_word_inserted_at_line_start(self):
        observations = [self.observation("cat", "a"), self.observation("the cat", "b")]
        result = decode_line(self.line(*observations), self.root, {}, FakeLexicon())
        self.assertEqual(result["ocr_text"], "cat")
        candidates = [item for menu in result["candidates"] for item in menu["items"]]
        self.assertTrue(any(item["text"] == "the cat" and item["origin"] == "ocr:b:full" for item in candidates))

    def test_overlapping_short_segments_are_not_duplicated(self):
        observations = [self.observation("l", anomalies=["short_read"]),
                        self.observation("hello world", variant="segment", segment_index=0, box=[0, 0, 65, 15]),
                        self.observation("world again", variant="segment", segment_index=1, box=[40, 0, 100, 15])]
        result = decode_line(self.line(*observations), self.root, {"mode": "visual"})
        self.assertEqual(result["text"], "hello world again")
        self.assertEqual(result["text"].count("world"), 1)

    def test_uncertain_segment_seam_preserves_full_reading(self):
        observations = [self.observation("old full reading"),
                        self.observation("hello earth", variant="segment", segment_index=0, box=[0, 0, 65, 15]),
                        self.observation("world again", variant="segment", segment_index=1, box=[40, 0, 100, 15])]
        result = decode_line(self.line(*observations), self.root, {"mode": "visual"})
        self.assertEqual(result["text"], "old full reading")
        self.assertIn("segment_alignment:a", result["needs_review"])

    def test_nonoverlapping_crops_cannot_be_joined_by_similar_words(self):
        parts = [dict(text="hello world", segment_index=0, box=[0, 0, 40, 15]),
                 dict(text="world again", segment_index=1, box=[50, 0, 100, 15])]
        self.assertIsNone(stitch_segments(parts))

    def test_corrupt_evidence_is_an_error(self):
        source = self.observation("text")
        (self.root / source["probabilities"]).write_bytes(b"wrong evidence")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            decode_line(self.line(source), self.root, {"mode": "visual"})


if __name__ == "__main__":
    unittest.main()
