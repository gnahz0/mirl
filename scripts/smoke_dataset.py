import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import datasets
from omegaconf import OmegaConf

import datasets as _d
_d.disable_progress_bars()

from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DATA = "data"
NAMES = ["smellnet", "ecg", "haptic_ts", "climb", "human_behaviour", "tactile"]
FILES = [os.path.join(DATA, f"{n}_train.parquet") for n in NAMES]

cfg = OmegaConf.create({
    "prompt_key": "prompt",
    "image_key": "images",
    "video_key": "videos",
    "max_prompt_length": 4096,
    "truncation": "left",
    "filter_overlong_prompts": False,
    "return_raw_chat": False,
    "return_multi_modal_inputs": True,
    "shuffle": False,
})

print(">> loading tokenizer/processor:", MODEL)
tok = hf_tokenizer(MODEL)
proc = hf_processor(MODEL, use_fast=True)

print(">> building RLHFDataset over", len(FILES), "parquet files")
ds = RLHFDataset(data_files=FILES, tokenizer=tok, config=cfg, processor=proc)
print(">> dataset len:", len(ds))

# offsets: first row of each dataset within the concatenation
offsets, acc = {}, 0
for n in NAMES:
    d = datasets.load_dataset("parquet", data_files=os.path.join(DATA, f"{n}_train.parquet"))["train"]
    offsets[n] = acc
    acc += len(d)

print("\n>> [1/2] __getitem__ returns raw_prompt (tokenization deferred to AgentLoop)")
ok = True
for n in NAMES:
    i = offsets[n]
    try:
        s = ds[i]
        assert "raw_prompt" in s
        print(f"  [{n:16}] idx={i:7d} OK  keys={sorted(s.keys())[:6]}")
    except Exception as e:
        ok = False
        print(f"  [{n:16}] idx={i:7d} FAILED: {type(e).__name__}: {e}")

print("\n>> [2/2] full multimodal path (_build_messages + apply_chat_template + process media + tokenize)")
import copy
from verl.utils.dataset.vision_utils import process_image, process_video
for n in NAMES:
    i = offsets[n]
    try:
        row = copy.deepcopy(ds.dataframe[i])
        n_img = len(row.get("images") or [])
        n_vid = len(row.get("videos") or [])
        messages = ds._build_messages(row)  # asserts placeholder count == media count
        prompt_str = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = [process_image(im, image_patch_size=ds.image_patch_size) for im in (row.get("images") or [])] or None
        videos = None
        if row.get("videos"):
            videos = [process_video(v, image_patch_size=ds.image_patch_size) for v in row["videos"]]
        out = proc(text=[prompt_str], images=images, videos=videos, return_tensors="pt")
        ntok = int(out["input_ids"].shape[-1])
        media = f"imgs={n_img}" if n_img else f"vids={n_vid}"
        print(f"  [{n:16}] idx={i:7d} OK  {media:9} tokens={ntok} processor_keys={sorted(out.keys())}")
    except Exception as e:
        ok = False
        import traceback
        print(f"  [{n:16}] idx={i:7d} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
