# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import torch

from mirl_ext.alignment.model import MultimodalAlignmentModel, temporal_crop


def _renderer() -> MultimodalAlignmentModel:
    model = MultimodalAlignmentModel.__new__(MultimodalAlignmentModel)
    torch.nn.Module.__init__(model)
    model.vit_patch_size = 16
    model.vit_merge_size = 2
    model.vit_temporal_patch_size = 2
    model.tactile_delta_channels = True
    return model


def _postmerge_tokens(grid_thw: torch.Tensor) -> int:
    return int((grid_thw.prod(dim=-1) // 4).sum())


def test_scalar_rendering_preserves_values_geometry_and_missing_masks():
    model = _renderer()
    captured = {}

    def capture(video):
        captured["video"] = video
        return video, torch.tensor([[video.shape[1] // 2, 2, 2]])

    model._patchify_pseudo_video = capture
    smell = torch.arange(40, dtype=torch.float32).unsqueeze(0)
    model._timeseries_to_video_inputs(smell)
    video = captured["video"][0]
    expected = model._robust_normalize_rows(smell)[0]
    assert torch.equal(video[0, 0, 0], expected[:32])
    assert torch.equal(video[1, 0, 0, :8], expected[32:])
    assert torch.equal(video[:, 0], video[:, 1])

    model = _renderer()
    ecg = torch.randn(8, 2500)
    ecg[3] = torch.nan
    pixels, grid = model._timeseries_to_video_inputs(ecg, "prestandardized")
    assert grid.tolist() == [[40, 16, 2]]
    assert _postmerge_tokens(grid) == 320
    assert torch.isfinite(pixels).all() and pixels.min() >= -1 and pixels.max() <= 1

    values = torch.tensor([[-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0]])
    expected = torch.tensor([[-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0]])
    assert torch.equal(model._normalize_scalar_rows(values, "prestandardized"), expected)


def test_temporal_crop_is_local_and_tactile_fields_stay_aligned():
    scalar = torch.arange(20, dtype=torch.float32).unsqueeze(0)
    crop = temporal_crop(scalar, "ecg", 6, generator=torch.Generator().manual_seed(3))
    start = int(crop[0, 0])
    assert crop.shape == (1, 6)
    assert torch.equal(crop, scalar[:, start : start + 6])

    pressure = torch.zeros(15, 16, 16)
    pressure[10] = 5.0
    payload = {
        "tactile": pressure,
        "force": torch.arange(15, dtype=torch.float32).unsqueeze(1),
    }
    crop = temporal_crop(payload, "tactile", 5)
    assert int(crop["force"][0]) == 8
    assert torch.equal(crop["tactile"], pressure[8:13])


def test_tactile_rendering_keeps_pressure_delta_and_force_cells_separate():
    model = _renderer()
    payload = {
        "tactile": torch.randn(47, 16, 16),
        "force": torch.randn(47, 13),
    }
    frames = model._tactile_frame_tiles(payload)
    pixels, grid = model._tactile_to_video_inputs(payload)

    assert frames.shape == (47, 3, 32, 64)
    assert torch.equal(frames[:, 0, :, :32], frames[:, 2, :, :32])
    assert grid.tolist() == [[24, 2, 4]]
    assert _postmerge_tokens(grid) == 48
    assert pixels.min() >= -1 and pixels.max() <= 1


def test_qwen_branch_can_return_tokens_or_pool_per_sample():
    model = _renderer()

    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.tokens = torch.eye(4)[:3]

        def forward(self, pixel_values, grid_thw):
            return SimpleNamespace(pooler_output=self.tokens)

    visual = FakeVisual()
    pixels = torch.zeros(1, 4)
    grid = torch.tensor([[1, 4, 2], [1, 2, 2]])
    tokens = model._encode_qwen_branch(pixels, grid, visual=visual, no_grad=False, pool=False)
    pooled = model._encode_qwen_branch(pixels, grid, visual=visual, no_grad=False, pool=True)

    assert torch.equal(tokens, visual.tokens)
    assert torch.equal(pooled, torch.stack((visual.tokens[:2].mean(0), visual.tokens[2])))
