import json
from pathlib import Path
import random
from typing import List, Literal

import fire
import pandas as pd


def select_pivots(
    quality_matrix_csv: str,
    output_path: str,
    top_k: int = 5,
    w_sp: float = 0.3,
    w_pt: float = 0.7,
    strategy: Literal["mpp", "english_only", "random"] = "mpp",
    seed: int = 42,
) -> List[dict]:
    """
    Stage 1: Pivot Selection (Section 3.1.1, Equation 1).

    Computes the pivot score for each candidate pivot language p not in {s, t}:
        S(s, p, t) = w_sp * Q(s, p) + w_pt * Q(p, t)
    with default weights w_sp = 0.3 and w_pt = 0.7, and selects the top-k pivots.

    Args:
        quality_matrix_csv: Path to pairwise COMET quality CSV Q(s, t) on FLORES-200 dev.
        output_path: Path to save selected pivots JSON.
        top_k: Number of candidate pivot languages to select per direction (default: 5).
        w_sp: Weight for source-to-pivot quality Q(s, p) (default: 0.3).
        w_pt: Weight for pivot-to-target quality Q(p, t) (default: 0.7).
        strategy: "mpp" (default quality-aware selection), "english_only", or "random".
        seed: Random seed used when strategy="random".
    """
    df = pd.read_csv(quality_matrix_csv, index_col=0)
    langs = df.columns.tolist()
    rng = random.Random(seed)
    results = []

    for src in langs:
        for trg in langs:
            if src == trg:
                continue

            if strategy == "english_only":
                candidate_pool = ["en"] if ("en" != src and "en" != trg and "en" in langs) else []
            else:
                candidate_pool = [p for p in langs if p != src and p != trg]

            pivot_scores = []
            for p in candidate_pool:
                try:
                    s1 = df.loc[src, p]  # Q(s, p): src -> pivot
                    s2 = df.loc[p, trg]  # Q(p, t): pivot -> trg
                    if pd.isna(s1) or pd.isna(s2):
                        continue
                    score = w_sp * float(s1) + w_pt * float(s2)
                    pivot_scores.append((p, float(s1), float(s2), float(score)))
                except KeyError:
                    continue

            if strategy == "random":
                selected = rng.sample(pivot_scores, min(top_k, len(pivot_scores)))
            elif strategy == "english_only":
                selected = pivot_scores[:1]
            else:
                pivot_scores.sort(key=lambda x: x[3], reverse=True)
                selected = pivot_scores[:top_k]

            top_pivots = [
                {
                    "pivot_lang": p,
                    "src_score": s1,
                    "trg_score": s2,
                    "final_score": score,
                }
                for p, s1, s2, score in selected
            ]

            results.append(
                {
                    "src_lang": src,
                    "trg_lang": trg,
                    "pivots": top_pivots,
                }
            )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"Saved pivot selections for {len(results)} directions "
        f"(strategy='{strategy}', top_k={top_k}, w_sp={w_sp}, w_pt={w_pt}) to {out_file}"
    )
    return results


if __name__ == "__main__":
    fire.Fire(select_pivots)
