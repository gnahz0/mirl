"""Script to save Qwen3.5 with a fine-tuned vision encoder. See README.md."""

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--vision-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.base_model)

    state = torch.load(
        args.vision_checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.model.visual.load_state_dict(
        state["trainable_visual"],
        strict=True,
    )

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()