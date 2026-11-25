import os
from typing import Optional, Sequence, Union

import fire

import eval.run_eval as run_eval
from mpp.prepare_flores import prepare_flores_data
from utils.configs import CORE_LANGS


def run(
    model_path: str,
    model_name: str = "",
    test_data_path: str = "flores200_dataset",
    split: str = "devtest",
    infer_data_path: Optional[str] = None,
    langs: Union[str, Sequence[str]] = CORE_LANGS,
    metrics: Union[str, Sequence[str]] = ("bleurt", "comet"),
    bleurt_path: str = "BLEURT-20",
    comet_path: str = "Unbabel/wmt22-comet-da",
    log_to_wandb: bool = False,
):
    """Prepare FLORES-200 parallel pairs and run Direct Translation + evaluation."""
    if not model_name:
        model_name = os.path.basename(os.path.normpath(model_path))

    lang_list = (
        [x.strip() for x in langs.split(",") if x.strip()]
        if isinstance(langs, str)
        else list(langs)
    )

    config = {
        "model_name": model_name,
        "langs": lang_list,
        "test_data_path": test_data_path,
        "metrics": metrics,
        "dataset": f"flores200-{split}",
        "split": split,
    }

    if infer_data_path is None:
        infer_data_path = f"outputs/{model_name}_flores200_{split}_direct.json"

    print(f"Preparing FLORES-200 ({split}) inference data at: {infer_data_path}")
    prepare_flores_data(
        flores_dir=test_data_path,
        output_path=infer_data_path,
        split=split,
        langs=lang_list,
    )

    run_eval.run(
        model_path=model_path,
        infer_data_path=infer_data_path,
        metrics=metrics,
        bleurt_path=bleurt_path,
        comet_path=comet_path,
        log_to_wandb=log_to_wandb,
        config=config,
    )


if __name__ == "__main__":
    fire.Fire(run)
