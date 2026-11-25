import os
import subprocess
import sys
import tempfile
from typing import Dict, List

import fire


def func_call(bleurt_path: str, mt_list: List[str], ref_list: List[str], scorer=None) -> Dict:
    from bleurt import score

    assert len(mt_list) == len(ref_list), "mt_list and ref_list lengths must match."

    if scorer is None:
        assert bleurt_path is not None, "bleurt_path must not be None."
        scorer = score.BleurtScorer(bleurt_path)

    scores = [
        float(s)
        for s in scorer.score(references=ref_list, candidates=mt_list, batch_size=128)
    ]
    system_score = sum(scores) / len(scores) if scores else 0.0
    print("bleurt:", system_score)

    return {"scores": scores, "system_score": system_score}


def cli_call(bleurt_path: str, mt_list: List[str], ref_list: List[str]) -> Dict:
    """Run BLEURT scoring in an isolated subprocess to release TensorFlow GPU memory afterwards."""
    assert len(mt_list) == len(ref_list), "mt_list and ref_list lengths must match."

    with tempfile.TemporaryDirectory() as tmpdir:
        mt_file = os.path.join(tmpdir, "candidates.mt")
        ref_file = os.path.join(tmpdir, "references.ref")
        out_file = os.path.join(tmpdir, "scores.out")

        with open(mt_file, "w", encoding="utf-8") as f_mt, open(
            ref_file, "w", encoding="utf-8"
        ) as f_ref:
            for mt, ref in zip(mt_list, ref_list):
                f_mt.write(mt.replace("\n", " ").strip() + "\n")
                f_ref.write(ref.replace("\n", " ").strip() + "\n")

        cmd = [
            sys.executable,
            "-m",
            "bleurt.score_files",
            f"-candidate_file={mt_file}",
            f"-reference_file={ref_file}",
            f"-bleurt_checkpoint={bleurt_path}",
            f"-scores_file={out_file}",
        ]
        subprocess.run(cmd, check=True)

        with open(out_file, "r", encoding="utf-8") as f_out:
            scores = [float(line.strip()) for line in f_out if line.strip()]

    system_score = sum(scores) / len(scores) if scores else 0.0
    print("bleurt:", system_score)
    return {"scores": scores, "system_score": system_score}


def main(
    mt_path: str,
    ref_path: str,
    output_path: str,
    bleurt_path: str = "BLEURT-20",
) -> Dict:
    with open(mt_path, "r", encoding="utf-8") as f_mt, open(
        ref_path, "r", encoding="utf-8"
    ) as f_ref:
        mt_list = [line.strip() for line in f_mt]
        ref_list = [line.strip() for line in f_ref]

    results = func_call(bleurt_path, mt_list, ref_list)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for s in results["scores"]:
            f_out.write(f"{s}\n")

    return results


if __name__ == "__main__":
    fire.Fire(main)
