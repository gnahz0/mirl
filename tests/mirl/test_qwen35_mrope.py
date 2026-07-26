import torch

from verl.experimental.agent_loop.agent_loop import _grid_aligned_mm_token_types


def test_timestamped_qwen3_video_grid_is_expanded_per_token_run():
    video_token = 7
    # Four timestamp-separated frame groups plus an unbacked generated token.
    input_ids = torch.tensor([[7, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 1, 7]])
    video_grid = torch.tensor([[4, 24, 24]])

    token_types, position_grid = _grid_aligned_mm_token_types(
        input_ids, None, video_token, None, video_grid
    )

    assert position_grid.tolist() == [[1, 24, 24]] * 4
    assert token_types.tolist() == [[2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 0]]


def test_legacy_contiguous_video_grid_is_not_expanded():
    input_ids = torch.tensor([[7, 7, 7, 1, 1]])
    video_grid = torch.tensor([[4, 24, 24]])

    token_types, position_grid = _grid_aligned_mm_token_types(input_ids, None, 7, None, video_grid)

    assert torch.equal(position_grid, video_grid)
    assert token_types.tolist() == [[2, 2, 2, 0, 0]]


def test_image_labels_stop_after_available_grids():
    input_ids = torch.tensor([[5, 5, 1, 5, 5, 1, 5]])
    image_grid = torch.tensor([[1, 12, 12], [1, 8, 8]])

    token_types, _ = _grid_aligned_mm_token_types(input_ids, 5, None, image_grid, None)

    assert token_types.tolist() == [[1, 1, 0, 1, 1, 0, 0]]
