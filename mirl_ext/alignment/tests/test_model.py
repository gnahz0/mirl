# Copyright 2026 Alec Zhang. Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import torch

from mirl_ext.alignment.model import MultimodalAlignmentModel, _enable_block_checkpointing


def _renderer() -> MultimodalAlignmentModel:
    model = MultimodalAlignmentModel.__new__(MultimodalAlignmentModel)
    torch.nn.Module.__init__(model)
    model.frame_side = 64
    model.max_frames = 512
    return model


def test_robust_normalization_uses_std_when_mad_is_zero():
    sparse = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    constant = torch.full((1, 4), 5.0)
    values = torch.cat((sparse, constant))

    normalized = MultimodalAlignmentModel._robust_normalize_rows(values)
    std = sparse.std(dim=-1, keepdim=True, unbiased=False)
    expected_sparse = torch.tanh(sparse / (2.0 * std))

    torch.testing.assert_close(normalized[0], expected_sparse[0])
    assert torch.equal(normalized[1], torch.zeros_like(normalized[1]))


def test_tactile_rendering_uses_four_spatial_tokens_and_1024_token_cap():
    model = _renderer()
    tactile = torch.randn(600, 16, 16)
    frames = model._tactile_frames(tactile[: model.max_frames])

    assert frames.shape == (512, 3, 64, 64)
    assert torch.equal(frames[:, 0], frames[:, 1])
    assert torch.equal(frames[:, 1], frames[:, 2])
    assert frames.min() >= -1 and frames.max() <= 1


def test_text_encoder_truncates_each_text_to_model_context():
    model = _renderer()

    class Tokens(dict):
        def to(self, device):
            return Tokens({key: value.to(device) for key, value in self.items()})

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["short", "long"]
            assert kwargs["max_length"] == 64
            assert kwargs["truncation"] is True
            assert "return_overflowing_tokens" not in kwargs
            return Tokens(input_ids=torch.tensor([[1.0], [2.0]]))

    class TextModel(torch.nn.Module):
        config = SimpleNamespace(max_position_embeddings=64)

        def forward(self, input_ids):
            return SimpleNamespace(pooler_output=torch.cat((input_ids, torch.zeros_like(input_ids)), dim=1))

    model.label_tokenizer = Tokenizer()
    model.label_text_model = TextModel()
    encoded = model.encode_text(["short", "long"], torch.device("cpu"))

    assert torch.equal(encoded, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_block_checkpointing_preserves_state_dict_keys():
    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

        def forward(self, value):
            return self.blocks[0](value)

    visual = Visual()
    keys = set(visual.state_dict())
    assert _enable_block_checkpointing(visual) == 1
    assert set(visual.state_dict()) == keys
    visual(torch.ones(1, 2, requires_grad=True)).sum().backward()
