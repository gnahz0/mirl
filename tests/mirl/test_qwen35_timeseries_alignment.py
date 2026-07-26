import torch

from mirl_ext.alignment.model import MultimodalAlignmentModel
from mirl_ext.data.clean_smellnet_indexes import canonical_label


def _renderer() -> MultimodalAlignmentModel:
    model = MultimodalAlignmentModel.__new__(MultimodalAlignmentModel)
    torch.nn.Module.__init__(model)
    model.vit_patch_size = 16
    model.vit_merge_size = 2
    model.vit_temporal_patch_size = 2
    return model


def _postmerge_tokens(grid_thw: torch.Tensor) -> int:
    return int((grid_thw.prod(dim=-1) // 4).sum())


def test_smellnet_label_canonicalization_removes_false_case_negatives():
    assert canonical_label("Almond80_Clove20") == "almond 80 clove 20"
    assert canonical_label("almond80_clove20") == "almond 80 clove 20"


def test_smell_native_shapes_respect_merger_boundaries():
    model = _renderer()

    pixels, grid = model._smell_to_pixel_inputs(torch.randn(4, 600))
    assert grid.tolist() == [[1, 8, 2]]
    assert _postmerge_tokens(grid) == 4
    assert pixels.shape[-1] == 3 * 2 * 16 * 16

    pixels, grid = model._smell_to_pixel_inputs(torch.randn(6, 867))
    assert grid.tolist() == [[1, 12, 2]]
    assert _postmerge_tokens(grid) == 6
    assert torch.isfinite(pixels).all()

    # Capacity expands instead of resizing once a sensor exceeds 32x32 values.
    _, grid = model._smell_to_pixel_inputs(torch.randn(4, 1025))
    assert grid.tolist() == [[1, 8, 4]]
    assert _postmerge_tokens(grid) == 8


def test_scalar_raster_is_direct_serpentine_values_not_a_plot():
    model = _renderer()
    captured = {}

    def capture(img):
        captured["img"] = img
        return img, torch.tensor([[1, img.shape[-2] // 16, img.shape[-1] // 16]])

    model._patchify_pseudo_image = capture
    signal = torch.arange(40, dtype=torch.float32).unsqueeze(0)
    model._timeseries_to_pixel_inputs(signal)
    image = captured["img"][0]
    expected = model._robust_normalize_rows(signal)[0]

    # First column runs down; the next reverses so t=31 and t=32 stay neighbors.
    assert torch.equal(image[0, :32, 0], expected[:32])
    assert torch.equal(image[0, 24:32, 1], expected[32:40].flip(0))
    assert torch.equal(image[0], image[1])
    assert torch.equal(image[1], image[2])
    # Exactly 40 positions are marked as real timesteps; padding remains -1.
    assert (image[2] > -1).sum().item() == 40


def test_smell_and_ecg_use_the_exact_same_scalar_renderer():
    model = _renderer()
    signal = torch.randn(3, 1100)
    smell_pixels, smell_grid = model._smell_to_pixel_inputs(signal)
    ecg_pixels, ecg_grid = model._ecg_to_pixel_inputs(signal)
    assert torch.equal(smell_pixels, ecg_pixels)
    assert torch.equal(smell_grid, ecg_grid)


def test_smell_video_uses_native_temporal_patch_pairs():
    model = _renderer()
    pixels, grid = model._smell_to_video_inputs(torch.randn(4, 600))
    # ceil(600 / 32) = 19 source frames, padded to 20, then fused in pairs.
    assert grid.tolist() == [[10, 8, 2]]
    assert _postmerge_tokens(grid) == 40
    assert torch.isfinite(pixels).all()


def test_tactile_video_emits_tactile_and_force_tokens_per_frame_pair():
    model = _renderer()
    tactile = {
        "tactile": torch.randn(47, 16, 16),
        "force": torch.randn(47, 13),
    }
    pixels, grid = model._tactile_to_video_inputs(tactile)
    assert grid.tolist() == [[24, 2, 4]]
    assert _postmerge_tokens(grid) == 48
    assert pixels.min() >= -1
    assert pixels.max() <= 1


def test_tactile_frames_are_grayscale_and_merger_cells_do_not_mix_semantics():
    model = _renderer()
    frames = model._tactile_frame_tiles({
        "tactile": torch.randn(5, 16, 16),
        "force": torch.randn(5, 13),
    })
    assert frames.shape == (5, 3, 32, 64)
    assert torch.equal(frames[:, 0], frames[:, 1])
    assert torch.equal(frames[:, 1], frames[:, 2])


def test_ecg_missing_lead_is_masked_not_propagated():
    model = _renderer()
    ecg = torch.randn(8, 2500)
    ecg[3] = torch.nan
    pixels, grid = model._ecg_to_pixel_inputs(ecg)
    assert grid.tolist() == [[1, 16, 6]]
    assert _postmerge_tokens(grid) == 24
    assert torch.isfinite(pixels).all()
    assert pixels.min() >= -1
    assert pixels.max() <= 1


def test_static_image_duplicates_temporal_plane_but_video_does_not():
    model = _renderer()
    image = torch.randn(1, 3, 32, 32)
    _, image_grid = model._patchify_pseudo_image(image)
    assert image_grid.tolist() == [[1, 2, 2]]

    video = torch.randn(1, 5, 3, 32, 32)
    _, video_grid = model._patchify_pseudo_video(video)
    # Odd fifth frame is repeated; six source frames become three temporal patches.
    assert video_grid.tolist() == [[3, 2, 2]]
