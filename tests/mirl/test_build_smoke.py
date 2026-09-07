"""The smoke fixture must represent the same closed families as production RL."""

from collections import Counter

import datasets
import pyarrow.parquet as pq

from mirl_ext.data.build_smoke import ACTIVE_SMOKE_SOURCES, SELECTIONS, _closed, build
from mirl_ext.data.schema import OPEN_SOURCES


def test_smoke_selection_matches_closed_production_mixture():
    assert SELECTIONS == {
        "ecg_train.parquet": 2,
        "climb_train.parquet": 2,
        "human_behaviour_train_closed.parquet": 2,
        "tactile_train_closed.parquet": 2,
    }
    assert sum(SELECTIONS.values()) == 8


def test_smoke_rejects_every_declared_open_source():
    assert not (ACTIVE_SMOKE_SOURCES & OPEN_SOURCES)
    assert all(not _closed({"data_source": source}) for source in OPEN_SOURCES)
    assert not _closed({"data_source": "unknown"})
    assert all(_closed({"data_source": source}) for source in ("ecg", "mri", "meld_emotion", "initial_fingers"))


def test_build_filters_open_and_unknown_rows_and_writes_eight_closed_rows(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"smoke-test-image")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"smoke-test-video")
    source_by_file = {
        "ecg_train.parquet": "ecg",
        "climb_train.parquet": "mri",
        "human_behaviour_train_closed.parquet": "meld_emotion",
        "tactile_train_closed.parquet": "initial_fingers",
    }

    for filename, closed_source in source_by_file.items():
        tactile = filename == "tactile_train_closed.parquet"
        rows = []
        for source, text in (
            ("intentqa", "a"),
            ("unknown", "b"),
            (closed_source, "closed one"),
            (closed_source, "closed response two"),
        ):
            rows.append(
                {
                    "data_source": source,
                    "prompt": [{"role": "user", "content": text}],
                    # Mirror the real cross-source schema mismatch: tactile has
                    # a null-only image list and min_frames field, while the
                    # other source files serialize concrete nested types.
                    "images": [] if tactile else [{"image": str(image)}],
                    "videos": [
                        {
                            "video": str(video),
                            "min_frames": None if tactile else 4,
                            "max_frames": 24 if tactile else 8,
                        }
                    ],
                }
            )
        # Dataset exports include source-specific HF feature metadata. The
        # merged fixture must discard it after unifying the Arrow schemas.
        datasets.Dataset.from_list(rows).to_parquet(tmp_path / filename)

    input_schemas = [pq.read_schema(tmp_path / filename) for filename in source_by_file]
    assert all(schema.metadata and b"huggingface" in schema.metadata for schema in input_schemas)
    assert input_schemas[0].field("images").type.value_type != input_schemas[-1].field("images").type.value_type
    assert (
        input_schemas[0].field("videos").type.value_type.field("min_frames").type
        != input_schemas[-1].field("videos").type.value_type.field("min_frames").type
    )

    output = tmp_path / "smoke.parquet"
    build(tmp_path, output, max_video_bytes=1024, scan_rows=16)

    result = pq.read_table(output)
    sources = result.column("data_source").to_pylist()
    assert result.num_rows == 8
    assert Counter(sources) == Counter({source: 2 for source in source_by_file.values()})
    assert set(sources) <= ACTIVE_SMOKE_SOURCES
    assert not result.schema.metadata

    hf_result = datasets.load_dataset("parquet", data_files=str(output), split="train")
    assert len(hf_result) == 8
    assert Counter(hf_result["data_source"]) == Counter({source: 2 for source in source_by_file.values()})
