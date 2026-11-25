import copy
import json
from pathlib import Path
from typing import List, Optional

import fire


def filter_pivots(
    input_path: str,
    output_path: Optional[str] = None,
    max_pivot: int = 15,
    top_k: int = 3,
    strict_greater: bool = False,
) -> List[dict]:
    """
    Stage 2 (QE Filtering): Filter low-quality pivot candidates using COMETKiwi-DA scores
    and retain at most top_k (default 3) candidates for Stage 3 aggregation
    (Section 3.1.2 & Section 3.1.3).

    Args:
        input_path: Path to JSON file with `direct_comet` and `pivot{i}_comet` scores.
        output_path: Output JSON path (defaults to overwriting `input_path`).
        max_pivot: Maximum pivot index to inspect (`pivot1` .. `pivot{max_pivot}`).
        top_k: Maximum number of retained pivot candidates per sample (default: 3).
        strict_greater: If True, require `pivot_comet > direct_comet`; otherwise `>=`.
    """
    if output_path is None:
        output_path = input_path

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing_direct_comet = 0
    fixed_direct_comet = 0
    skipped_samples = 0
    fallback_direct_count = 0
    filtered_data = []

    for sample in data:
        sample = copy.deepcopy(sample)

        if "direct_comet" not in sample or sample["direct_comet"] is None:
            missing_direct_comet += 1
            direct_txt = (
                sample.get("direct_target_text")
                or sample.get("direct_text")
                or sample.get("mt_text")
                or ""
            )
            if str(direct_txt).strip() == "":
                sample["direct_comet"] = 0.0
                fixed_direct_comet += 1
            else:
                skipped_samples += 1
                continue

        direct_score = float(sample["direct_comet"])

        # Step 1: Collect valid non-empty pivot candidates beating direct translation
        valid_pivots = []
        for i in range(1, max_pivot + 1):
            comet_key = f"pivot{i}_comet"
            text_key = f"pivot{i}_target_text"
            pivot_text = str(sample.get(text_key) or "").strip()
            pivot_score = sample.get(comet_key)

            if not pivot_text or pivot_score is None:
                continue

            pivot_score = float(pivot_score)
            passes = (
                pivot_score > direct_score
                if strict_greater
                else pivot_score >= direct_score
            )
            if passes:
                valid_pivots.append((i, pivot_score))

        # Step 2: Sort retained candidates by QE score descending
        valid_pivots.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Keep only top-K pivots
        keep_indices = {i for i, _ in valid_pivots[:top_k]}
        if not keep_indices:
            fallback_direct_count += 1

        # Step 4: Remove all pivots not in top-K
        for i in range(1, max_pivot + 1):
            if i not in keep_indices:
                keys_to_delete = [
                    f"pivot{i}_lang",
                    f"pivot{i}_text",
                    f"pivot{i}_target_text",
                    f"pivot{i}_comet",
                    f"pivot{i}_processed_tokens",
                ]
                for key in keys_to_delete:
                    sample.pop(key, None)

        filtered_data.append(sample)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Samples missing 'direct_comet': {missing_direct_comet}")
    print(f"Samples fixed with empty direct text: {fixed_direct_comet}")
    print(f"Samples skipped: {skipped_samples}")
    print(f"Samples falling back to direct translation (0 retained pivots): {fallback_direct_count}")
    print(f"Final samples saved: {len(filtered_data)} -> {out_file}")
    return filtered_data


if __name__ == "__main__":
    fire.Fire(filter_pivots)
