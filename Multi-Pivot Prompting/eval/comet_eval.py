import os
from typing import Dict, List, Optional

from comet import download_model, load_from_checkpoint
import fire


def load_comet_model(comet_path: str):
    """Load a COMET or COMETKiwi checkpoint from a local path or HuggingFace model ID."""
    if os.path.exists(comet_path):
        ckpt_path = comet_path
    else:
        ckpt_path = download_model(comet_path)
    return load_from_checkpoint(ckpt_path)


def func_call(
    comet_path: str,
    src_list: List[str],
    mt_list: List[str],
    ref_list: Optional[List[str]] = None,
    model=None,
    batch_size: int = 256,
    gpus: int = 1,
    num_workers: int = 4,
) -> Dict:
    """
    Evaluate translations with COMET (reference-based if `ref_list` is given,
    or reference-free QE mode if `ref_list` is None).
    """
    assert len(src_list) == len(mt_list), "src_list and mt_list lengths must match."

    use_ref = ref_list is not None
    if use_ref:
        assert len(src_list) == len(ref_list), "src_list and ref_list lengths must match."

    data = []
    if use_ref:
        for src, mt, ref in zip(src_list, mt_list, ref_list):
            data.append({"src": src.strip(), "mt": mt.strip(), "ref": ref.strip()})
    else:
        for src, mt in zip(src_list, mt_list):
            data.append({"src": src.strip(), "mt": mt.strip()})

    if model is None:
        model = load_comet_model(comet_path)

    model_output = model.predict(
        data,
        batch_size=batch_size,
        gpus=gpus,
        num_workers=num_workers,
    )

    print("comet:", model_output.system_score)

    return {
        "scores": [float(s) for s in model_output.scores],
        "metadata": getattr(model_output, "metadata", None),
        "system_score": float(model_output.system_score),
    }


def main(
    comet_path: str,
    src_path: str,
    mt_path: str,
    ref_path: Optional[str] = None,
) -> Dict:
    with open(src_path, "r", encoding="utf-8") as f_src, open(
        mt_path, "r", encoding="utf-8"
    ) as f_mt:
        src_list = [line.strip() for line in f_src]
        mt_list = [line.strip() for line in f_mt]

    ref_list = None
    if ref_path:
        with open(ref_path, "r", encoding="utf-8") as f_ref:
            ref_list = [line.strip() for line in f_ref]

    return func_call(comet_path, src_list, mt_list, ref_list)


if __name__ == "__main__":
    fire.Fire(main)
