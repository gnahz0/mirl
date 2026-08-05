# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import torch
import torch.utils.checkpoint as torch_checkpoint

from mirl_ext.alignment.model import MultimodalAlignmentModel, _enable_block_checkpointing


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


def test_tactile_uses_fixed_opentouch_pressure_scale():
    values = torch.tensor([[-10.0, 0.0, 1536.0, 3072.0, 4000.0]])
    expected = torch.tensor([[-1.0, -1.0, 0.0, 1.0, 1.0]])
    assert torch.equal(MultimodalAlignmentModel._normalize_tactile(values), expected)


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


def test_text_encoder_exposes_and_mean_pools_all_overflow_chunks():
    model = _renderer()

    class Tokens(dict):
        def to(self, device):
            return Tokens({key: value.to(device) for key, value in self.items()})

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["short", "long"]
            assert kwargs["max_length"] == 64
            assert kwargs["return_overflowing_tokens"] is True
            return Tokens(
                input_ids=torch.tensor([[1.0], [2.0], [4.0]]),
                overflow_to_sample_mapping=torch.tensor([0, 1, 1]),
            )

    class TextModel(torch.nn.Module):
        config = SimpleNamespace(max_position_embeddings=64)

        def forward(self, input_ids):
            return SimpleNamespace(pooler_output=torch.cat((input_ids, torch.zeros_like(input_ids)), dim=1))

    model.label_tokenizer = Tokenizer()
    model.label_text_model = TextModel()
    chunks, owners = model.encode_text_chunks(["short", "long"], torch.device("cpu"))
    encoded = model.encode_text(["short", "long"], torch.device("cpu"))

    assert torch.equal(chunks, torch.tensor([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
    assert torch.equal(owners, torch.tensor([0, 1, 1]))
    assert torch.equal(encoded, torch.tensor([[1.0, 0.0], [3.0, 0.0]]))


def test_block_checkpointing_is_training_only(monkeypatch):
    calls = []

    class Block(torch.nn.Module):
        def forward(self, value, scale=1):
            return value * scale

    block = Block()
    visual = SimpleNamespace(blocks=torch.nn.ModuleList([block]))

    def checkpoint(function, *args, use_reentrant, **kwargs):
        calls.append(use_reentrant)
        return function(*args, **kwargs)

    monkeypatch.setattr(torch_checkpoint, "checkpoint", checkpoint)
    assert _enable_block_checkpointing(visual) == 1

    block.train()
    assert block(torch.tensor(2.0), scale=3).item() == 6
    assert calls == [False]

    block.eval()
    assert block(torch.tensor(2.0), scale=4).item() == 8
    assert calls == [False]
