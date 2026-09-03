from types import SimpleNamespace

import pytest
import torch

from mirl_ext.alignment.model import AlignmentOutput, MultimodalAlignmentModel
from mirl_ext.data.signals import family_frames, normalize, signal_family, tactile_frames, timeseries_frames


def _renderer() -> MultimodalAlignmentModel:
    model = MultimodalAlignmentModel.__new__(MultimodalAlignmentModel)
    torch.nn.Module.__init__(model)
    model.max_video_frames = 8
    model.tactile_video_fps = 1.0
    model.tactile_min_video_frames = 4
    model.tactile_max_video_frames = 24
    return model


def test_scalar_rendering_preserves_values_geometry_and_missing_masks():
    signal = torch.arange(40, dtype=torch.float32).unsqueeze(0)
    video = timeseries_frames(signal, 32)
    expected = normalize(signal)[0]
    assert torch.equal(video[0, 0, 0], expected[:32])
    assert torch.equal(video[1, 0, 0, :8], expected[32:])
    assert torch.equal(video[:, 0], video[:, 1])

    ecg = torch.randn(8, 2500)
    frames = timeseries_frames(ecg, 32)
    assert frames.shape == (79, 3, 256, 32)
    assert torch.isfinite(frames).all() and frames.min() >= -1 and frames.max() <= 1

    values = torch.tensor([[-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0]])
    expected = (values / values.std(dim=-1, keepdim=True, unbiased=False)).clamp(-4.0, 4.0) / 4.0
    frames = timeseries_frames(values, 32)
    torch.testing.assert_close(frames[0, 0, 0, :7], expected[0])


def test_normalization_zscores_rows_and_zeroes_constant():
    sparse = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    constant = torch.full((1, 4), 5.0)
    values = torch.cat((sparse, constant))

    normalized = normalize(values)
    mean = sparse.mean(dim=-1, keepdim=True)
    std = sparse.std(dim=-1, keepdim=True, unbiased=False)
    expected_sparse = ((sparse - mean) / std).clamp(-4.0, 4.0) / 4.0

    torch.testing.assert_close(normalized[0], expected_sparse[0])
    assert torch.equal(normalized[1], torch.zeros_like(normalized[1]))
    assert normalized.min() >= -1 and normalized.max() <= 1


def test_normalization_rejects_non_finite_rows():
    with pytest.raises(ValueError, match="non-finite"):
        normalize(torch.tensor([[1.0, float("nan"), 3.0]]))


def test_tactile_rendering_uses_one_rgb_cell_per_full_frame():
    tactile = torch.randn(47, 16, 16)
    frames = tactile_frames(tactile, 32)

    assert frames.shape == (47, 3, 32, 32)
    assert torch.equal(frames[:, 0], frames[:, 1])
    assert torch.equal(frames[:, 1], frames[:, 2])
    assert frames.min() >= -1 and frames.max() <= 1


def test_alignment_output_is_named_and_tuple_compatible():
    tensors = [torch.tensor(float(index)) for index in range(6)]
    output = AlignmentOutput(*tensors)

    assert output.visual_embeddings is tensors[0]
    assert output.reference_embeddings is tensors[1]
    assert output.visual_token_counts is tensors[2]
    assert output.signal_embeddings is tensors[3]
    assert tuple(output) == tuple(tensors)


def test_signal_and_media_dispatches_fail_closed():
    with pytest.raises(ValueError, match="unsupported signal format"):
        signal_family({"format": "unknown"})
    with pytest.raises(ValueError, match="unsupported signal family"):
        family_frames(torch.zeros(1, 4), "unknown", 32)
    with pytest.raises(ValueError, match="unsupported alignment media kind"):
        _renderer().forward("audio", [torch.zeros(1)], None, 1024)
    with pytest.raises(ValueError, match="missing its data source"):
        _renderer().forward("video", ["clip.mp4"], None, 1024)
    with pytest.raises(ValueError, match="empty media batch"):
        _renderer().forward("image", [], None, 1024)
    with pytest.raises(ValueError, match="channels, time"):
        timeseries_frames(torch.empty(8, 0), 32)
    with pytest.raises(ValueError, match="time, height, width"):
        tactile_frames(torch.empty(0, 16, 16), 32)


def test_video_sampling_is_tactile_specific():
    model = _renderer()
    assert model.video_sampling_kwargs("initial_fingers") == {
        "fps": 1.0,
        "num_frames": None,
        "min_frames": 4,
        "max_frames": 24,
    }
    expected_fixed_eight = {
        "fps": None,
        "num_frames": None,
        "min_frames": 1,
        "max_frames": 8,
    }
    assert model.video_sampling_kwargs("mosei_emotion") == expected_fixed_eight
    assert model.video_sampling_kwargs("climb") == expected_fixed_eight


def test_qwen_branch_can_return_tokens_or_pool_per_sample():
    model = _renderer()

    class FakeVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.tokens = torch.eye(4)[:3]

        def forward(self, pixel_values, grid_thw):
            return SimpleNamespace(last_hidden_state=self.tokens)

    visual = FakeVisual()
    pixels = torch.zeros(1, 4)
    grid = torch.tensor([[1, 2, 1], [1, 1, 1]])
    model.trainable_visual = visual
    tokens = model.encode_visual(pixels, grid, pool=False)
    pooled = model.encode_visual(pixels, grid, pool=True)

    assert torch.equal(tokens, visual.tokens)
    assert torch.equal(pooled, torch.stack((visual.tokens[:2].mean(0), visual.tokens[2])))


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
