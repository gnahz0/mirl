#!/usr/bin/env python3
"""Verify the pinned MIRL Qwen3.5 environment without downloading artifacts."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


EXPECTED_VERSIONS = {
    "torch": "2.10.0+cu129",
    "torchvision": "0.25.0+cu129",
    "torchaudio": "2.10.0+cu129",
    "transformers": "5.3.0.dev0",
    "vllm": "0.18.0",
    "flash-attn": "2.8.3",
    "flash-linear-attention": "0.5.1",
    "causal-conv1d": "1.6.2.post1",
    "nvidia-cutlass-dsl": "4.4.2",
    "quack-kernels": "0.3.4",
    "qwen-vl-utils": "0.0.14",
    "torchcodec": "0.10.0",
    "TransferQueue": "0.1.8",
    "verl": "0.9.0.dev0",
}
EXPECTED_REVISIONS = {
    "transformers": "cc7ab9be508ce6ed3637bba9e50367b29b742dc6",
    "flash-linear-attention": "c525f4957f11a6f197b52c0c222377446c3eab56",
}
VERL_BASE_REVISION = "6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"
EXPECTED_PIP_CHECK_ISSUE = (
    "vllm 0.18.0 has requirement transformers<5,>=4.56.0, "
    "but you have transformers 5.3.0.dev0."
)


def _verify_versions() -> None:
    assert sys.version_info[:2] == (3, 12), sys.version
    for package, expected in EXPECTED_VERSIONS.items():
        actual = metadata.version(package)
        assert actual == expected, f"{package}: expected {expected}, got {actual}"
        print(f"{package}={actual}")


def _verify_revisions() -> None:
    for package, expected in EXPECTED_REVISIONS.items():
        raw = metadata.distribution(package).read_text("direct_url.json")
        assert raw, f"{package}: direct_url.json is missing"
        commit = json.loads(raw)["vcs_info"]["commit_id"]
        assert commit == expected, f"{package}: expected {expected}, got {commit}"
        print(f"{package}_revision={commit}")


def _verify_imports() -> None:
    import flash_attn
    import fla
    import causal_conv1d
    import cutlass
    import cutlass.cute
    import qwen_vl_utils
    import ray
    import torch
    import torchcodec
    import transfer_queue
    import transformers
    import verl
    import vllm
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.registry import ModelRegistry

    modules = (
        causal_conv1d,
        flash_attn,
        fla,
        qwen_vl_utils,
        ray,
        torch,
        torchcodec,
        transfer_queue,
        transformers,
        verl,
        vllm,
    )
    for module in modules:
        source = Path(module.__file__).resolve()
        assert not str(source).startswith("/home/"), f"user-site import: {source}"

    assert Qwen3_5Config.model_type == "qwen3_5"
    assert Qwen3_5ForConditionalGeneration.__name__ == "Qwen3_5ForConditionalGeneration"
    assert "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs()
    assert hasattr(cutlass.cute.core, "ThrMma"), "CUTLASS DSL is too new for vLLM 0.18"

    cuda_include = Path(sys.prefix) / "targets" / "x86_64-linux" / "include"
    for header in ("cublasLt.h", "nvrtc.h"):
        assert (cuda_include / header).is_file(), f"CUDA JIT header is missing: {header}"

    import vllm.vllm_flash_attn.cute.interface  # noqa: F401

    from qwen_vl_utils.vision_process import get_video_reader_backend

    assert get_video_reader_backend() == "torchcodec"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        sibling_ffmpeg = Path(sys.executable).with_name("ffmpeg")
        ffmpeg = str(sibling_ffmpeg) if sibling_ffmpeg.is_file() else None
    assert ffmpeg, "ffmpeg executable is missing"
    subprocess.run([ffmpeg, "-version"], check=True, stdout=subprocess.DEVNULL)
    print(f"video_backend={get_video_reader_backend()}")
    print(f"ffmpeg={ffmpeg}")

    verl_root = Path(verl.__file__).resolve().parents[1]
    subprocess.run(
        ["git", "-C", str(verl_root), "merge-base", "--is-ancestor", VERL_BASE_REVISION, "HEAD"],
        check=True,
    )
    print(f"verl_checkout={verl_root}")


def _verify_pip_check() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    issues = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert issues == [EXPECTED_PIP_CHECK_ISSUE], f"unexpected pip check output: {issues}"
    print("pip_check=known_vllm_transformers_metadata_mismatch_only")


def _verify_snapshot(snapshot: Path) -> None:
    from transformers import AutoConfig, AutoProcessor

    assert snapshot.is_dir(), snapshot
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    assert config.model_type == "qwen3_5"
    assert config.vision_config.patch_size == 16
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Classify this signal."}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    print(f"model_snapshot={snapshot}")
    print("vision_patch_size=16")
    print("default_thinking_template=True")


def _verify_cuda() -> None:
    import torch
    from causal_conv1d import causal_conv1d_fn
    from flash_attn import flash_attn_func
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "NVIDIA B200"

    qkv = torch.randn((1, 64, 4, 64), device="cuda", dtype=torch.bfloat16)
    flash_output = flash_attn_func(qkv, qkv, qkv, causal=True)
    assert torch.isfinite(flash_output).all()

    conv_input = torch.randn((1, 8, 64), device="cuda", dtype=torch.bfloat16)
    conv_weight = torch.randn((8, 4), device="cuda", dtype=torch.bfloat16)
    conv_output = causal_conv1d_fn(conv_input, conv_weight, activation="silu")
    assert torch.isfinite(conv_output).all()

    shape = (1, 16, 2, 32)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand((1, 16, 2), device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn((1, 16, 2), device="cuda", dtype=torch.float32))
    fla_output, final_state = fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        output_final_state=True,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(fla_output).all()
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"flash_attn_output={tuple(flash_output.shape)}")
    print(f"causal_conv1d_output={tuple(conv_output.shape)}")
    print(f"fla_output={tuple(fla_output.shape)} state={tuple(final_state.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="run B200 CUDA kernel checks")
    parser.add_argument("--model-snapshot", type=Path, help="verify a local Qwen3.5 snapshot")
    args = parser.parse_args()

    _verify_versions()
    _verify_revisions()
    _verify_imports()
    _verify_pip_check()
    if args.model_snapshot:
        _verify_snapshot(args.model_snapshot)
    if args.cuda:
        _verify_cuda()


if __name__ == "__main__":
    main()
