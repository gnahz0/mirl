"""Tests for tactile video QA reward scoring."""

from verl.utils.reward_score.tactile import (
    acc_reward,
    compute_score,
    extract_boxed_answer,
    f1_score,
    format_reward,
    jaccard_similarity,
    parse_labels,
)


def test_extract_boxed_answer():
    assert extract_boxed_answer("blah \\boxed{A}") == "A"
    assert extract_boxed_answer("\\boxed{B,C}") == "B,C"
    assert extract_boxed_answer("no boxed here") is None
    # nested braces
    assert extract_boxed_answer("\\boxed{A{x}}") == "A{x}"
    # picks last boxed
    assert extract_boxed_answer("\\boxed{A} then \\boxed{B}") == "B"


def test_parse_labels():
    assert parse_labels("A") == {"A"}
    assert parse_labels("B,C,D") == {"B", "C", "D"}
    assert parse_labels("a, b") == {"A", "B"}
    assert parse_labels("") == set()
    assert parse_labels(None) == set()


def test_format_reward():
    assert format_reward("<think>reasoning</think>\n\\boxed{A}") == 1.0
    assert format_reward("<think>reasoning</think>\\boxed{A}") == 1.0
    assert format_reward("\\boxed{A}") == 0.0  # no think tags
    assert format_reward("<think></think>\\boxed{A}") == 1.0
    assert format_reward("just text") == 0.0


def test_acc_reward():
    assert acc_reward({"A"}, {"A"}) == 1.0
    assert acc_reward({"A", "B"}, {"B", "A"}) == 1.0  # order agnostic
    assert acc_reward({"A"}, {"B"}) == 0.0
    assert acc_reward({"A", "B"}, {"A"}) == 0.0


def test_jaccard_similarity():
    assert jaccard_similarity({"A"}, {"A"}) == 1.0
    assert jaccard_similarity({"A", "B"}, {"B", "C"}) == 1 / 3
    assert jaccard_similarity({"A"}, {"B"}) == 0.0
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({"A"}, set()) == 0.0


def test_f1_score():
    p, r, f = f1_score({"A", "B"}, {"A", "B"})
    assert (p, r, f) == (1.0, 1.0, 1.0)

    p, r, f = f1_score({"A", "B", "C"}, {"A", "B"})
    assert r == 1.0
    assert abs(p - 2 / 3) < 1e-9

    p, r, f = f1_score({"A"}, {"A", "B"})
    assert p == 1.0
    assert r == 0.5

    p, r, f = f1_score(set(), set())
    assert (p, r, f) == (1.0, 1.0, 1.0)

    p, r, f = f1_score({"A"}, set())
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_compute_score_single_label():
    # Correct answer with proper format
    result = compute_score("<think>ok</think>\\boxed{A}", "A")
    assert result["acc"] == 1.0
    assert result["jaccard"] == 1.0
    assert result["f1"] == 1.0
    assert result["format"] == 1.0
    assert result["score"] == 0.5 * 1 + 0.4 * 1 + 0.1 * 1  # 1.0

    # Wrong answer, good format
    result = compute_score("<think>hmm</think>\\boxed{B}", "A")
    assert result["acc"] == 0.0
    assert result["jaccard"] == 0.0
    assert result["format"] == 1.0
    assert result["score"] == 0.1  # only format score


def test_compute_score_multi_label():
    # Exact match multi-label
    result = compute_score("<think>ok</think>\\boxed{A,B}", "A,B")
    assert result["acc"] == 1.0
    assert result["jaccard"] == 1.0

    # Order agnostic
    result = compute_score("<think>ok</think>\\boxed{B,A}", "A,B")
    assert result["acc"] == 1.0

    # Partial overlap
    result = compute_score("<think>ok</think>\\boxed{A,C}", "A,B")
    assert result["acc"] == 0.0
    assert abs(result["jaccard"] - 1 / 3) < 1e-9
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_compute_score_no_boxed():
    result = compute_score("I think the answer is A", "A")
    assert result["acc"] == 0.0
    assert result["format"] == 0.0
    assert result["score"] == 0.0


def test_compute_score_all_numeric():
    """All returned values should be float (no None/str) for metric_utils compatibility."""
    result = compute_score("no answer", "A")
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is {type(val)}, expected float"
