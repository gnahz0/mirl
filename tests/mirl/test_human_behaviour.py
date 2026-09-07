"""Stdlib-only regressions for HBA's closed-classification reward adapter."""

import unittest

from mirl_ext.rewards.human_behaviour import compute_score


class HumanBehaviourRewardTests(unittest.TestCase):
    def test_correct_formatted_label_gets_full_reward(self):
        result = compute_score(r"<think>Reasoning.</think>\boxed{sad}", "sad")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["format"], 1.0)

    def test_opposite_categories_get_no_label_or_similarity_credit(self):
        for prediction, target in (
            ("ptsd", "no ptsd"),
            ("sarcasm", "not sarcasm"),
            ("humour", "not humour"),
            ("weakly positive", "weakly negative"),
            ("sad", "surprise"),
        ):
            with self.subTest(prediction=prediction, target=target):
                result = compute_score("<think>Reason.</think>\\boxed{" + prediction + "}", target)
                self.assertEqual(result["score"], 0.2)
                self.assertEqual(result["format"], 1.0)
                for key in ("acc", "precision", "recall", "f1", "jaccard", "similarity"):
                    self.assertEqual(result[key], 0.0, key)

    def test_whitespace_and_case_are_normalized_for_both_labels(self):
        result = compute_score("<think>Reason.</think>\\boxed{ \tNo PTSD\n}", "\tNO PTSD \n")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_internal_whitespace_and_punctuation_are_not_fuzzy_matched(self):
        for prediction in ("no  ptsd", "no ptsd."):
            with self.subTest(prediction=prediction):
                self.assertEqual(compute_score(prediction, "no ptsd")["acc"], 0.0)

    def test_first_box_wins(self):
        response = r"<think>Reason.</think>\boxed{sad} then \boxed{happy}"
        self.assertEqual(compute_score(response, "sad")["score"], 1.0)
        self.assertEqual(compute_score(response, "happy")["score"], 0.2)

    def test_box_takes_priority_over_earlier_fallback_notations(self):
        response = r"[happy] <answer>anger</answer> \boxed{sad}"
        self.assertEqual(compute_score(response, "sad")["score"], 0.8)
        self.assertEqual(compute_score(response, "happy")["acc"], 0.0)

    def test_brackets_then_answer_then_full_text_fallbacks(self):
        for response in ("[sad]", "<answer>sad</answer>", " sad "):
            with self.subTest(response=response):
                result = compute_score(response, "sad")
                self.assertEqual(result["score"], 0.8)
                self.assertEqual(result["acc"], 1.0)
                self.assertEqual(result["format"], 0.0)
        self.assertEqual(compute_score("<answer>happy</answer> [sad]", "sad")["acc"], 1.0)
        self.assertEqual(compute_score("[sad] [happy]", "sad")["acc"], 1.0)

    def test_malformed_box_can_fall_back_to_brackets(self):
        result = compute_score(r"\boxed{oops [sad]", "sad")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["format"], 0.0)

    def test_tag_spacing_is_normalized_before_scoring(self):
        response = r"< think > Reason. < / think > \boxed{sad}"
        self.assertEqual(compute_score(response, "sad")["score"], 1.0)
        self.assertEqual(compute_score("< answer > sad < / answer >", "sad")["score"], 0.8)

    def test_no_think_prefix_is_invented_by_the_task_scorer(self):
        # Qwen prompt-prefix restoration belongs to the combined adapter.
        result = compute_score(r"Reason.</think>\boxed{sad}", "sad")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["format"], 0.0)

    def test_missing_or_wrong_answer_without_format_gets_zero(self):
        for response in ("", "anger", "<think>Reason.</think>"):
            with self.subTest(response=response):
                self.assertEqual(compute_score(response, "sad")["score"], 0.0)

    def test_empty_target_keeps_upstream_exact_match_semantics(self):
        self.assertEqual(compute_score("", "")["score"], 0.8)
        self.assertEqual(compute_score(r"<think>Reason.</think>\boxed{}", " ")["score"], 1.0)

    def test_long_response_has_no_inner_length_penalty(self):
        response = "<think>" + "reason " * 10000 + "</think>\\boxed{sad}"
        result = compute_score(response, "sad")
        self.assertEqual(result["score"], 1.0)
        self.assertNotIn("overlong_score", result)

    def test_shared_metric_contract_contains_only_numeric_values(self):
        keys = {"score", "acc", "precision", "recall", "f1", "jaccard", "similarity", "format"}
        result = compute_score(r"<think>Reason.</think>\boxed{no ptsd}", "no ptsd")
        self.assertEqual(set(result), keys)
        for key, value in result.items():
            self.assertIsInstance(value, float, key)
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)
        self.assertEqual(result["similarity"], result["jaccard"])


if __name__ == "__main__":
    unittest.main()
