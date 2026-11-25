import json
from pathlib import Path
from typing import Optional

import fire

import eval.comet_eval as comet_eval


def score_candidates_qe(
    input_path: str,
    output_path: Optional[str] = None,
    comet_qe_path: str = "Unbabel/wmt22-cometkiwi-da",
    max_pivot: int = 15,
    batch_size: int = 256,
    gpus: int = 1,
) -> str:
    """
    Stage 2 (QE Scoring): Reference-free Quality Estimation with COMETKiwi-DA
    (Section 3.1.2).

    Computes reference-free QE scores for:
      - Direct translation (`direct_target_text`, `direct_text`, or `mt_text`) -> `direct_comet`
      - Pivot-to-target candidates (`pivot{i}_target_text`) -> `pivot{i}_comet`

    Args:
        input_path: Path to inference JSON file containing direct and pivot target candidates.
        output_path: Output JSON path (defaults to overwriting `input_path`).
        comet_qe_path: Local COMETKiwi-DA `.ckpt` file path or HuggingFace model ID
                       (default: "Unbabel/wmt22-cometkiwi-da").
        max_pivot: Maximum pivot index to scan (`pivot1` .. `pivot{max_pivot}`).
        batch_size: Batch size for COMET prediction.
        gpus: Number of GPUs for COMET prediction.
    """
    if output_path is None:
        output_path = input_path

    with open(input_path, "r", encoding="utf-8") as f:
        infer_data = json.load(f)

    # Load COMETKiwi-DA once and reuse across direct + all pivots
    qe_model = comet_eval.load_comet_model(comet_qe_path)

    # 1. Score pivot candidates (reference-free)
    for pivot_num in range(1, max_pivot + 1):
        pivot_mt_key = f"pivot{pivot_num}_target_text"
        pivot_score_key = f"pivot{pivot_num}_comet"

        src_texts = []
        mt_texts = []
        idx_map = []

        for idx, item in enumerate(infer_data):
            if pivot_mt_key not in item:
                continue
            src = (item.get("src_text") or "").strip()
            mt = (item.get(pivot_mt_key) or "").strip()

            if src and mt:
                src_texts.append(src)
                mt_texts.append(mt)
                idx_map.append(idx)
            else:
                item[pivot_score_key] = 0.0

        if not src_texts:
            continue

        print(f"Scoring {pivot_score_key} ({len(src_texts)} samples)...")
        out = comet_eval.func_call(
            comet_path=comet_qe_path,
            src_list=src_texts,
            mt_list=mt_texts,
            ref_list=None,
            model=qe_model,
            batch_size=batch_size,
            gpus=gpus,
        )
        for i, idx in enumerate(idx_map):
            infer_data[idx][pivot_score_key] = float(out["scores"][i])

    # 2. Score direct translation candidate (reference-free)
    direct_src = []
    direct_mt = []
    idx_map = []

    for idx, item in enumerate(infer_data):
        src = (item.get("src_text") or "").strip()
        # Prefer direct_target_text, then direct_text, then mt_text
        raw_direct = (
            item.get("direct_target_text")
            if "direct_target_text" in item
            else item.get("direct_text", item.get("mt_text", ""))
        )
        mt = (raw_direct or "").strip()

        if src and mt:
            direct_src.append(src)
            direct_mt.append(mt)
            idx_map.append(idx)
        elif any(k in item for k in ("direct_target_text", "direct_text", "mt_text")):
            item["direct_comet"] = 0.0

    if direct_src:
        print(f"Scoring direct_comet ({len(direct_src)} samples)...")
        out = comet_eval.func_call(
            comet_path=comet_qe_path,
            src_list=direct_src,
            mt_list=direct_mt,
            ref_list=None,
            model=qe_model,
            batch_size=batch_size,
            gpus=gpus,
        )
        for i, idx in enumerate(idx_map):
            infer_data[idx]["direct_comet"] = float(out["scores"][i])

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(infer_data, f, ensure_ascii=False, indent=2)

    print(f"Saved reference-free COMETKiwi-DA QE scores to {out_file}")
    return str(out_file)


if __name__ == "__main__":
    fire.Fire(score_candidates_qe)
