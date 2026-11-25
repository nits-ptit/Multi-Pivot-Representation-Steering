import json
from itertools import permutations
from pathlib import Path
from typing import List, Optional, Sequence, Union

import fire

from utils.configs import CORE_LANGS, flores_langcode_map


def prepare_flores_data(
    flores_dir: str,
    output_path: str,
    split: str = "devtest",
    langs: Optional[Union[str, Sequence[str]]] = None,
    target_lang: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    Prepare all-pairs parallel translation JSON from the FLORES-200 dataset.

    Args:
        flores_dir: Root directory of FLORES-200 containing `{split}/{flores_code}.{split}`
                    (or directory directly containing `{flores_code}.{split}`).
        output_path: Output JSON file path.
        split: Dataset split ("dev" for pivot quality estimation, "devtest" for evaluation).
        langs: List or comma-separated string of ISO-2 language codes (defaults to the 10 paper languages).
        target_lang: Optional single target language ISO-2 code (e.g., "zh") to build only X -> target_lang pairs.
        max_samples: Optional cap on number of sentences per language pair.
    """
    if langs is None:
        lang_list = list(CORE_LANGS)
    elif isinstance(langs, str):
        lang_list = [x.strip() for x in langs.split(",") if x.strip()]
    else:
        lang_list = list(langs)

    base_path = Path(flores_dir)
    split_dir = base_path / split if (base_path / split).is_dir() else base_path

    lang_data = {}
    for iso2 in lang_list:
        if iso2 not in flores_langcode_map:
            raise ValueError(f"Unsupported ISO-2 language code: {iso2}")
        flores_code = flores_langcode_map[iso2]

        candidates = [
            split_dir / f"{flores_code}.{split}",
            split_dir / f"{split}.{flores_code}",
            split_dir / flores_code,
        ]
        file_path = next((p for p in candidates if p.exists()), None)
        if file_path is None:
            raise FileNotFoundError(
                f"Missing FLORES-200 file for '{iso2}' ({flores_code}) in {split_dir}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            if max_samples is not None:
                lines = lines[:max_samples]
            lang_data[iso2] = lines

    lengths = [len(v) for v in lang_data.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Sentence count mismatch across FLORES files: {lengths}")

    num_samples = lengths[0]
    dataset = []

    for src_lang, trg_lang in permutations(lang_list, 2):
        if target_lang is not None and trg_lang != target_lang:
            continue

        src_sentences = lang_data[src_lang]
        trg_sentences = lang_data[trg_lang]
        for i in range(num_samples):
            dataset.append(
                {
                    "src_lang": src_lang,
                    "trg_lang": trg_lang,
                    "src_text": src_sentences[i],
                    "trg_text": trg_sentences[i],
                }
            )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(
        f"Saved {len(dataset)} samples ({len(dataset) // max(num_samples, 1)} directions, "
        f"split='{split}') to {out_file}"
    )
    return dataset


if __name__ == "__main__":
    fire.Fire(prepare_flores_data)
