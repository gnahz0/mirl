"""Strict ECG category correctness; diagnostics do not shape the reward."""

import unittest

from mirl_ext.rewards import ecg


class ECGRewardTests(unittest.TestCase):
    def test_all_categories_accept_case_and_whitespace(self):
        for category in ecg.CATEGORIES:
            answer = " \n" + category.upper().replace(" ", "\t  ") + "  "
            with self.subTest(category=category):
                self.assertEqual(ecg._predicted_category(answer), category)
                self.assertEqual(ecg.compute_score(answer, category)["score"], 1.0)
                self.assertEqual(ecg.compute_score(r"\boxed{" + answer + "}", category)["score"], 1.0)

    def test_prose_substrings_negations_and_multiple_labels_are_rejected(self):
        answers = [
            "Abnormal",
            "Otherness",
            "Not Normal",
            "Not Normal.",
            "Normal.",
            "Result: Normal",
            "Normal, Other",
            "Normal and Other",
            "Normal&Other",
            "<think>Normal</think>",
            "",
            "   ",
        ]
        for answer in answers:
            for prediction in (answer, r"<think>x</think>\boxed{" + answer + "}"):
                with self.subTest(prediction=prediction):
                    result = ecg.compute_score(prediction, "Normal")
                    self.assertEqual(result["score"], 0.0)
                    self.assertEqual(result["acc"], 0.0)
        self.assertEqual(ecg.compute_score(r"\boxed{Normal", "Normal")["score"], 0.0)

    def test_last_boxed_answer_is_authoritative(self):
        self.assertEqual(ecg.compute_score(r"\boxed{Normal} then \boxed{Other}", "Other")["score"], 1.0)
        self.assertEqual(ecg.compute_score(r"\boxed{Normal} then \boxed{Other}", "Normal")["score"], 0.0)
        self.assertEqual(ecg.compute_score(r"Other in reasoning; \boxed{Normal}", "Normal")["score"], 1.0)
        self.assertEqual(ecg.compute_score(r"Normal in reasoning; \boxed{}", "Normal")["score"], 0.0)

    def test_format_is_diagnostic_only(self):
        cases = [
            ("Normal", 1.0, 0.0),
            (r"\boxed{Normal}", 1.0, 0.0),
            (r"<think>x</think>\boxed{Normal}", 1.0, 1.0),
            (r"<think>x</think>\boxed{Other}", 0.0, 1.0),
        ]
        for prediction, score, fmt in cases:
            with self.subTest(prediction=prediction):
                result = ecg.compute_score(prediction, "Normal")
                self.assertEqual(result["score"], score)
                self.assertEqual(result["format"], fmt)

    def test_diagnostics_compare_whole_categories_not_words(self):
        for prediction in ("Conduction", "Myocardial Infarction", "Not Conduction Disturbance"):
            with self.subTest(prediction=prediction):
                result = ecg.compute_score(prediction, "Conduction Disturbance")
                self.assertEqual(
                    result,
                    dict.fromkeys(
                        ("score", "acc", "precision", "recall", "f1", "jaccard", "similarity", "format"),
                        0.0,
                    ),
                )
        correct = ecg.compute_score("Conduction Disturbance", "Conduction Disturbance")
        self.assertEqual(len(correct), 8)
        self.assertEqual(correct["f1"], 1.0)
        self.assertEqual(correct["jaccard"], 1.0)

    def test_invalid_ground_truth_fails_closed(self):
        for target in ("", " ", "Abnormal", "Normal.", "Normal, Other", None):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    ecg.compute_score("Normal", target)

    def test_ground_truth_accepts_case_and_whitespace(self):
        self.assertEqual(ecg.compute_score("Conduction Disturbance", " CONDUCTION \t DISTURBANCE ")["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
