from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import fire

import eval.bleurt_eval as bleurt_eval
import eval.comet_eval as comet_eval
from utils.configs import valid_langs
from utils.helpers import flat_text

MPP_ROOT = Path(__file__).resolve().parent.parent


def run_vllm_inference(
    model_path: str,
    infer_data_path: str,
    add_target_reference: bool = False,
    target_reference_prompt: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int = 768,
    chat_template: bool = True,
    reuse_direct_fallback: bool = False,
    save_prompt: bool = False,
):
    """Run vLLM generation in a subprocess so GPU VRAM is freed before metric evaluation."""
    abs_infer_path = os.path.abspath(infer_data_path)
    cmd = [
        sys.executable,
        "-m",
        "inference.run_vllm_mt_generation",
        "--model_path",
        str(model_path),
        "--infer_data_path",
        abs_infer_path,
        "--add_target_reference",
        str(add_target_reference),
        "--target_reference_prompt",
        str(target_reference_prompt),
        "--temperature",
        str(temperature),
        "--max_new_tokens",
        str(max_new_tokens),
        "--chat_template",
        str(chat_template),
        "--reuse_direct_fallback",
        str(reuse_direct_fallback),
        "--save_prompt",
        str(save_prompt),
    ]
    subprocess.run(cmd, check=True, cwd=str(MPP_ROOT))


def _group_data_aggregation(grouped_data: dict, metrics: List[str]) -> dict:
    lang_pair_scores = {}
    for pair, data in grouped_data.items():
        lang_pair_scores[pair] = {}
        if "bleurt" in metrics and data["bleurt_scores"]:
            lang_pair_scores[pair]["bleurt"] = sum(data["bleurt_scores"]) / len(
                data["bleurt_scores"]
            )
        if "comet" in metrics and data["comet_scores"]:
            lang_pair_scores[pair]["comet"] = sum(data["comet_scores"]) / len(
                data["comet_scores"]
            )
    return lang_pair_scores


def run_evaluation(infer_data_path: str, metrics: List[str], **kwargs) -> dict:
    """
    Evaluate `mt_text` against `trg_text` using BLEURT and/or COMET across all language pairs.
    Writes sentence-level metric scores back into `infer_data_path` without dropping samples.
    """
    with open(infer_data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    valid_indices = [
        idx
        for idx, item in enumerate(all_data)
        if item.get("mt_text") not in (None, "")
    ]

    if not valid_indices:
        print("No examples with 'mt_text' found. Skipping metric evaluation.")
        return {}

    valid_items = [all_data[idx] for idx in valid_indices]

    if "bleurt" in metrics:
        full_mt_list = [flat_text(item["mt_text"]) for item in valid_items]
        full_ref_list = [flat_text(item["trg_text"]) for item in valid_items]

        bleurt_path = kwargs.get("bleurt_path")
        if not bleurt_path:
            raise ValueError("`bleurt_path` must be provided when 'bleurt' is in metrics.")

        bleurt_out = bleurt_eval.cli_call(bleurt_path, full_mt_list, full_ref_list)
        if len(bleurt_out["scores"]) != len(valid_items):
            raise ValueError(
                f"BLEURT score count ({len(bleurt_out['scores'])}) != sample count ({len(valid_items)})."
            )
        for idx, score in zip(valid_indices, bleurt_out["scores"]):
            all_data[idx]["bleurt"] = float(score)

    if "comet" in metrics:
        full_src_list = [flat_text(item["src_text"]) for item in valid_items]
        full_mt_list = [flat_text(item["mt_text"]) for item in valid_items]
        full_ref_list = [flat_text(item["trg_text"]) for item in valid_items]

        comet_path = kwargs.get("comet_path")
        if not comet_path:
            raise ValueError("`comet_path` must be provided when 'comet' is in metrics.")

        comet_out = comet_eval.func_call(
            comet_path, full_src_list, full_mt_list, full_ref_list
        )
        if len(comet_out["scores"]) != len(valid_items):
            raise ValueError(
                f"COMET score count ({len(comet_out['scores'])}) != sample count ({len(valid_items)})."
            )
        for idx, score in zip(valid_indices, comet_out["scores"]):
            all_data[idx]["comet"] = float(score)

    grouped_data = defaultdict(
        lambda: {"mt": [], "ref": [], "bleurt_scores": [], "comet_scores": []}
    )

    for idx in valid_indices:
        item = all_data[idx]
        pair = (item["src_lang"], item["trg_lang"])
        grouped_data[pair]["mt"].append(flat_text(item["mt_text"]))
        grouped_data[pair]["ref"].append(flat_text(item["trg_text"]))
        if "bleurt" in metrics and "bleurt" in item:
            grouped_data[pair]["bleurt_scores"].append(item["bleurt"])
        if "comet" in metrics and "comet" in item:
            grouped_data[pair]["comet_scores"].append(item["comet"])

    lang_pair_scores = _group_data_aggregation(grouped_data, metrics)

    with open(infer_data_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    return lang_pair_scores


def log_results(
    lang_pair_scores: Dict[tuple, Dict[str, float]],
    config: Dict[str, Any],
    metrics: List[str],
    log_to_wandb: bool = False,
    langs: List[str] = valid_langs,
):
    overall_scores = {}
    for metric in metrics:
        metric_scores = [
            scores[metric]
            for scores in lang_pair_scores.values()
            if metric in scores and scores[metric] is not None
        ]
        if metric_scores:
            overall_scores[metric] = sum(metric_scores) / len(metric_scores)

    if log_to_wandb:
        log_results_to_wandb(lang_pair_scores, config, metrics, langs, overall_scores)
    else:
        log_results_to_stdout(lang_pair_scores, config, metrics, langs, overall_scores)


def log_results_to_wandb(
    lang_pair_scores: Dict[tuple, Dict[str, float]],
    config: Dict[str, Any],
    metrics: List[str],
    langs: List[str],
    overall_scores: Dict[str, float],
):
    import wandb

    wandb.init(
        project="mpp-eval-matrix",
        name=f"{config['model_name']}",
        config=config,
    )

    for metric in metrics:
        if not any(metric in s for s in lang_pair_scores.values()):
            continue

        columns = ["source_lang \\ target_lang"] + langs
        matrix_rows = []
        for src in langs:
            row: List[Any] = [src]
            for trg in langs:
                score = lang_pair_scores.get((src, trg), {}).get(metric, None)
                row.append(round(score, 4) if isinstance(score, float) else score)
            matrix_rows.append(row)

        wandb.log(
            {f"score_matrix/{metric}": wandb.Table(columns=columns, data=matrix_rows)}
        )

    if overall_scores:
        overall_data = [[m, round(s, 4)] for m, s in overall_scores.items()]
        wandb.log(
            {
                "overall_performance": wandb.Table(
                    columns=["Metric", "Overall Score (avg over lang pairs)"],
                    data=overall_data,
                )
            }
        )

    wandb.finish()


def log_results_to_stdout(
    lang_pair_scores: Dict[tuple, Dict[str, float]],
    config: Dict[str, Any],
    metrics: List[str],
    langs: List[str],
    overall_scores: Dict[str, float],
):
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {config['model_name']}")
    print("=" * 80)

    if overall_scores:
        print("\nOVERALL PERFORMANCE (average over language pairs):")
        print("-" * 50)
        for metric, score in overall_scores.items():
            print(f"{metric.upper():>12}: {score:.4f} ({score * 100:.2f})")

    for metric in metrics:
        if not any(metric in s for s in lang_pair_scores.values()):
            continue

        print(f"\n{metric.upper()} SCORE MATRIX:")
        print("-" * 50)
        corner_label = "src\\tgt"
        header = f"{corner_label:>8}" + "".join(f"{trg:>8}" for trg in langs)
        print(header)

        for src in langs:
            row = f"{src:>8}"
            for trg in langs:
                score = lang_pair_scores.get((src, trg), {}).get(metric, None)
                if isinstance(score, float):
                    row += f"{score:>8.4f}"
                elif score is None:
                    row += f"{'N/A':>8}"
                else:
                    row += f"{str(score):>8}"
            print(row)

    print("\n" + "=" * 80)


def run(
    model_path: str,
    infer_data_path: str,
    metrics: Union[str, Sequence[str]] = ("bleurt", "comet"),
    bleurt_path: str = "BLEURT-20",
    comet_path: str = "Unbabel/wmt22-comet-da",
    log_to_wandb: bool = False,
    config: Optional[dict] = None,
    add_target_reference: bool = False,
    target_reference_prompt: bool = False,
    skip_inference: bool = False,
    skip_evaluation: bool = False,
    temperature: float = 0.0,
    max_new_tokens: int = 768,
    chat_template: bool = True,
    reuse_direct_fallback: bool = False,
    save_prompt: bool = False,
    **kwargs,
):
    if isinstance(metrics, str):
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    else:
        metric_list = list(metrics)

    config = dict(config or {})
    if "model_name" not in config:
        config["model_name"] = os.path.basename(os.path.normpath(model_path))

    if not os.path.exists(infer_data_path):
        raise ValueError(f"Inference data path '{infer_data_path}' does not exist.")

    required_fields = ["src_lang", "trg_lang", "src_text"]
    with open(infer_data_path, "r", encoding="utf-8") as f:
        infer_data = json.load(f)
    for item in infer_data:
        if not all(field in item for field in required_fields):
            raise ValueError(
                f"Missing required fields {required_fields} in item: {item}"
            )

    if not skip_inference:
        print("Running model inference...")
        run_vllm_inference(
            model_path=model_path,
            infer_data_path=infer_data_path,
            add_target_reference=add_target_reference,
            target_reference_prompt=target_reference_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            chat_template=chat_template,
            reuse_direct_fallback=reuse_direct_fallback,
            save_prompt=save_prompt,
        )

    # Stage 2 (`add_target_reference=True`) generates intermediate pivot_target_text, not final mt_text
    if add_target_reference or skip_evaluation:
        print(f"Finished candidate generation for {config['model_name']}.")
        return

    print("Running metric evaluation...")
    lang_pair_scores = run_evaluation(
        infer_data_path,
        metric_list,
        bleurt_path=bleurt_path,
        comet_path=comet_path,
    )

    if lang_pair_scores:
        log_results(
            lang_pair_scores,
            config,
            metric_list,
            log_to_wandb=log_to_wandb,
        )
    print(f"Finished run for {config['model_name']}.")


if __name__ == "__main__":
    fire.Fire(run)
