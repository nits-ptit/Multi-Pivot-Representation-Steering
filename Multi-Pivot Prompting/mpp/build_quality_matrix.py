from collections import defaultdict
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import fire
import pandas as pd

from utils.configs import CORE_LANGS


def build_quality_matrix(
    dev_direct_path: str,
    output_csv_path: str,
    metric_key: str = "comet",
    langs: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Build the empirical pairwise translation quality matrix Q(s, t) on FLORES-200 dev
    (Section 3.1.1) from evaluated direct translation results.

    Args:
        dev_direct_path: JSON file with evaluated direct translations on FLORES-200 dev
                         (containing `src_lang`, `trg_lang`, and `comet` scores).
        output_csv_path: Output CSV path (rows = source language, cols = target language).
        metric_key: Metric field name in JSON items (default: "comet").
        langs: Optional language ordering (defaults to languages present in data / CORE_LANGS).
    """
    with open(dev_direct_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pair_scores = defaultdict(list)
    discovered_langs = []

    for item in data:
        src = item["src_lang"]
        trg = item["trg_lang"]
        if src not in discovered_langs:
            discovered_langs.append(src)
        if trg not in discovered_langs:
            discovered_langs.append(trg)

        score = item.get(metric_key)
        if isinstance(score, (int, float)):
            pair_scores[(src, trg)].append(float(score))

    if not pair_scores:
        raise ValueError(
            f"No numeric '{metric_key}' scores found in {dev_direct_path}. "
            f"Run COMET evaluation on the dev direct translations first."
        )

    if langs is None:
        ordered_langs = [l for l in CORE_LANGS if l in discovered_langs]
        for l in discovered_langs:
            if l not in ordered_langs:
                ordered_langs.append(l)
    elif isinstance(langs, str):
        ordered_langs = [x.strip() for x in langs.split(",") if x.strip()]
    else:
        ordered_langs = list(langs)

    rows = []
    for src in ordered_langs:
        row = {}
        for trg in ordered_langs:
            if src == trg:
                row[trg] = None
            else:
                scores = pair_scores.get((src, trg), [])
                row[trg] = round(sum(scores) / len(scores), 4) if scores else None
        rows.append(row)

    df = pd.DataFrame(rows, index=ordered_langs)
    df.index.name = "src_lang"

    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, na_rep="N/A")
    print(f"Saved pairwise quality matrix ({len(ordered_langs)}x{len(ordered_langs)}) to {out_path}")
    return df


if __name__ == "__main__":
    fire.Fire(build_quality_matrix)
