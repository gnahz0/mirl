"""Upstream MIRL's condition-set F1 reward with unchanged exact-match diagnostics."""

import unittest

from mirl_ext.rewards import medical


class MedicalRewardTests(unittest.TestCase):
    def test_normalizes_all_separator_styles_case_and_whitespace(self):
        target = "Edema, Pneumonia, Atelectasis, Pleural Effusion"
        predictions = [
            "  EDEMA,  Pneumonia&Atelectasis and Pleural \n Effusion ",
            "Pleural Effusion & Atelectasis,Pneumonia and Edema",
            "Edema, Pneumonia, Atelectasis, Pleural Effusion, Edema",
        ]
        for answer in predictions:
            with self.subTest(answer=answer):
                result = medical.compute_score(r"\boxed{" + answer + "}", target)
                self.assertEqual(result["score"], 0.5)
                for metric in ("acc", "precision", "recall", "f1", "jaccard", "similarity"):
                    self.assertEqual(result[metric], 1.0)
                self.assertEqual(result["format"], 0.0)

    def test_exact_reordering_has_identical_reward_and_diagnostics(self):
        target = "Pleural Effusion, Support Devices"
        forward = medical.compute_score(r"<think>x</think>\boxed{" + target + "}", target)
        reverse = medical.compute_score(r"<think>x</think>\boxed{Support Devices, Pleural Effusion}", target)
        self.assertEqual(forward, reverse)
        self.assertEqual(reverse["score"], 0.5)
        self.assertEqual(reverse["jaccard"], 1.0)

    def test_partial_match_gets_upstream_half_f1_reward_not_exact_accuracy(self):
        result = medical.compute_score(r"<think>x</think>\boxed{Edema}", "Edema, Pneumonia")
        self.assertAlmostEqual(result["score"], 1 / 3)
        self.assertEqual(result["acc"], 0.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertEqual(result["jaccard"], 0.5)
        self.assertEqual(result["similarity"], 0.5)
        self.assertEqual(result["format"], 1.0)

    def test_extra_unknown_condition_reduces_reward_and_is_not_exact(self):
        result = medical.compute_score(r"<think>x</think>\boxed{Edema, Unknown Diagnosis}", "Edema")
        self.assertAlmostEqual(result["score"], 1 / 3)
        self.assertEqual(result["acc"], 0.0)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)

    def test_boxed_answer_required_and_format_has_no_bonus(self):
        cases = [
            ("Edema", 0.0, 0.0),
            (r"\boxed{Edema}", 0.5, 0.0),
            (r"<think>x</think>\boxed{Edema}", 0.5, 1.0),
            (r"<think>x</think>\boxed{Pneumonia}", 0.0, 1.0),
            (r"<think>x</think>\boxed{}", 0.0, 1.0),
        ]
        for prediction, score, fmt in cases:
            with self.subTest(prediction=prediction):
                result = medical.compute_score(prediction, "Edema")
                self.assertEqual(len(result), 8)
                self.assertEqual(result["score"], score)
                self.assertEqual(result["format"], fmt)

    def test_last_boxed_answer_is_authoritative(self):
        self.assertEqual(medical.compute_score(r"\boxed{Edema} then \boxed{Pneumonia}", "Edema")["score"], 0.0)
        self.assertEqual(medical.compute_score(r"\boxed{Pneumonia} then \boxed{Edema}", "Edema")["score"], 0.5)

    def test_length_and_bbox_json_do_not_add_reward(self):
        plain = medical.compute_score(r"<think>x</think>\boxed{Edema}", "Edema")
        verbose = "<think>" + "reasoning " * 100 + "</think>"
        verbose += '\n```json\n[{"bbox_2d": [0, 0, 10, 10], "label": "Edema"}]\n```\n'
        verbose += r"\boxed{Edema}"
        self.assertEqual(medical.compute_score(verbose, "Edema"), plain)

    def test_empty_or_invalid_ground_truth_fails_closed(self):
        for target in ("", " ", "\n\t", " , & and ", None):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    medical.compute_score(r"\boxed{Edema}", target)

    def test_separator_words_do_not_split_inside_condition_words(self):
        self.assertEqual(
            medical.parse_conditions("Hand edema and Mandibular Lesion"), {"hand edema", "mandibular lesion"}
        )


if __name__ == "__main__":
    unittest.main()
