import json
from pathlib import Path
from typing import List, Optional

import fire

from utils.configs import flores_langcode_map


def construct_pivot_data(
    pivot_selection_path: str,
    output_path: str,
    flores_dir: Optional[str] = None,
    split: str = "devtest",
    base_data_path: Optional[str] = None,
    direct_mt_path: Optional[str] = None,
    target_lang: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    Construct the MPP candidate dataset JSON by attaching selected pivot languages
    and (if available) precomputed direct translations `s -> t` and first-step
    pivot translations `s -> p_i` from `direct_mt_path`.

    Args:
        pivot_selection_path: Path to pivot selection JSON from `mpp.pivot_selection`.
        output_path: Output path for constructed pivot JSON.
        flores_dir: FLORES-200 dataset directory (used if `base_data_path` is not provided).
        split: FLORES split ("devtest" for test evaluation, "dev" for MPRS pseudo-target generation).
        base_data_path: Optional pre-built JSON dataset with `src_lang`, `trg_lang`, `src_text`, `trg_text`.
        direct_mt_path: Optional JSON containing direct translations (`mt_text`) across language pairs,
                        used to populate `direct_target_text` (s -> t) and `pivot{i}_text` (s -> p_i).
        target_lang: Optional target language filter (e.g., "ko", "zh", "it").
        max_samples: Optional cap on sentences per direction when reading FLORES.
    """
    with open(pivot_selection_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if target_lang is not None:
        pairs = [row for row in pairs if row["trg_lang"] == target_lang]

    pivot_map = {(row["src_lang"], row["trg_lang"]): row["pivots"] for row in pairs}

    # Build or load base samples
    output = []
    if base_data_path is not None:
        with open(base_data_path, "r", encoding="utf-8") as f:
            base_samples = json.load(f)

        for sample in base_samples:
            src = sample["src_lang"]
            trg = sample["trg_lang"]
            if target_lang is not None and trg != target_lang:
                continue
            if (src, trg) not in pivot_map:
                continue

            item = {
                "src_lang": src,
                "trg_lang": trg,
                "src_text": sample["src_text"],
                "trg_text": sample.get("trg_text", ""),
            }
            for idx, p in enumerate(pivot_map[(src, trg)], start=1):
                item[f"pivot{idx}_lang"] = p["pivot_lang"]
            output.append(item)
    else:
        if flores_dir is None:
            raise ValueError("Either `flores_dir` or `base_data_path` must be provided.")

        base_path = Path(flores_dir)
        split_dir = base_path / split if (base_path / split).is_dir() else base_path

        # Collect all languages referenced in pivot selections
        needed_langs = set()
        for row in pairs:
            needed_langs.add(row["src_lang"])
            needed_langs.add(row["trg_lang"])
            for p in row["pivots"]:
                needed_langs.add(p["pivot_lang"])

        flores_data = {}
        for iso2 in sorted(needed_langs):
            flores_code = flores_langcode_map.get(iso2)
            if flores_code is None:
                continue
            candidates = [
                split_dir / f"{flores_code}.{split}",
                split_dir / f"{split}.{flores_code}",
                split_dir / flores_code,
            ]
            file_path = next((p for p in candidates if p.exists()), None)
            if file_path is not None:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines()]
                    if max_samples is not None:
                        lines = lines[:max_samples]
                    flores_data[iso2] = lines

        for row in pairs:
            src = row["src_lang"]
            trg = row["trg_lang"]
            if src not in flores_data or trg not in flores_data:
                continue

            pivots = row["pivots"]
            num_samples = min(len(flores_data[src]), len(flores_data[trg]))
            for i in range(num_samples):
                item = {
                    "src_lang": src,
                    "trg_lang": trg,
                    "src_text": flores_data[src][i],
                    "trg_text": flores_data[trg][i],
                }
                for idx, p in enumerate(pivots, start=1):
                    pivot_lang = p["pivot_lang"]
                    if pivot_lang in flores_data:
                        item[f"pivot{idx}_lang"] = pivot_lang
                output.append(item)

    # Map precomputed direct translations (s -> t and s -> p_i) if provided
    if direct_mt_path is not None:
        with open(direct_mt_path, "r", encoding="utf-8") as f:
            mt_data = json.load(f)

        mt_lookup = {}
        for item in mt_data:
            key = (item["src_lang"], item["trg_lang"], item["src_text"])
            mt_lookup[key] = item.get("mt_text", None)

        for sample in output:
            src_lang = sample["src_lang"]
            trg_lang = sample["trg_lang"]
            src_text = sample["src_text"]

            direct_key = (src_lang, trg_lang, src_text)
            if mt_lookup.get(direct_key) is not None:
                sample["direct_target_text"] = mt_lookup[direct_key]

            for i in range(1, 16):
                pivot_lang = sample.get(f"pivot{i}_lang")
                if pivot_lang:
                    pivot_key = (src_lang, pivot_lang, src_text)
                    if mt_lookup.get(pivot_key) is not None:
                        sample[f"pivot{i}_text"] = mt_lookup[pivot_key]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Constructed {len(output)} pivot samples -> {out_file}")
    return output


if __name__ == "__main__":
    fire.Fire(construct_pivot_data)
