import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

# IMPORTANT: set spawn start method before importing vLLM
mp.set_start_method("spawn", force=True)

import fire
from vllm import LLM, SamplingParams

from utils.helpers import post_clean
from utils.prompts import (
    DIRECT_SYSTEM_PROMPT,
    get_eval_prompt_direct_og,
    get_eval_target_reference_prompt,
)


def _format_chat_prompt(tokenizer, messages: List[dict], enable_thinking: bool = False) -> str:
    """Apply tokenizer chat template while disabling thinking tokens when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _build_direct_input(
    tokenizer,
    src_text: str,
    src_lang: str,
    trg_lang: str,
    chat_template: bool = True,
) -> str:
    prompt = get_eval_prompt_direct_og(src_lang, trg_lang).format(src_text)
    if chat_template:
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return _format_chat_prompt(tokenizer, messages, enable_thinking=False)
    return prompt


def generate_batch(model: LLM, input_texts: List[str], sampling_params: SamplingParams):
    outputs = model.generate(input_texts, sampling_params)

    clean_outputs = []
    processed_tokens = []

    for output in outputs:
        clean_outputs.append(post_clean(output.outputs[0].text))
        processed_tokens.append(
            len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
        )

    return clean_outputs, processed_tokens


def func_call(
    model: LLM,
    tokenizer,
    src_list: List[str],
    src_langs,
    trg_langs,
    chat_template: bool = True,
    max_new_tokens: int = 768,
    sampling_params: Optional[SamplingParams] = None,
    temperature: float = 0.0,
) -> dict:
    """Batch direct translation helper."""
    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)

    if sampling_params is None:
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

    if src_langs and trg_langs:
        example_prompt = get_eval_prompt_direct_og(src_langs[0], trg_langs[0])
        print(f"Using direct translation prompt:\n{example_prompt}")

    input_texts = []
    for src_text, src_lang, trg_lang in zip(src_list, src_langs, trg_langs):
        input_texts.append(
            _build_direct_input(
                tokenizer=tokenizer,
                src_text=src_text,
                src_lang=src_lang,
                trg_lang=trg_lang,
                chat_template=chat_template,
            )
        )

    responses, processed_tokens = generate_batch(model, input_texts, sampling_params)
    return {
        "response": responses,
        "prompt": input_texts,
        "processed_tokens": processed_tokens,
    }


def main(
    model_path: str,
    infer_data_path: str,
    out_data_path: Optional[str] = None,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    chat_template: bool = True,
    add_target_reference: bool = False,
    target_reference_prompt: bool = False,
    max_pivot: int = 15,
    reuse_direct_fallback: bool = False,
    save_prompt: bool = False,
    **kwargs,
):
    """
    Run vLLM translation generation for Multi-Pivot Prompting (MPP).

    Modes:
      1) Direct Translation (default: all mode flags False):
         Translates `src_text` (src_lang -> trg_lang) into `mt_text`.
      2) Stage 2 - Pivot Candidate Generation (`--add_target_reference True`):
         - Automatically generates missing `direct_target_text` (s -> t) and `pivot{i}_text` (s -> p_i)
           if they were not pre-populated from all-pairs direct translations.
         - Translates `pivot{i}_text` (pivot{i}_lang -> trg_lang) into `pivot{i}_target_text`.
      3) Stage 3 - Candidate Aggregation (`--target_reference_prompt True`):
         Aggregates top QE-filtered `pivot{i}_target_text` references with their quality scores
         using the Multi-Pivot Prompting template (Appendix A.2), falling back to Direct
         Translation when no pivot candidate remains after filtering.
    """
    with open(infer_data_path, "r", encoding="utf-8") as f:
        infer_data = json.load(f)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
    )

    # -------------------------------------------------------
    # Mode 1: Stage 2 - Pivot Candidate Generation (s->p & p->t)
    # -------------------------------------------------------
    if add_target_reference:
        print("Running MPP Stage 2: Pivot Candidate Generation...")

        # 1a. Populate direct_target_text (s -> t) if missing
        missing_direct_indices = [
            idx
            for idx, sample in enumerate(infer_data)
            if sample.get("direct_target_text") in (None, "")
        ]
        if missing_direct_indices:
            print(
                f"Generating missing direct_target_text for {len(missing_direct_indices)} samples..."
            )
            direct_inputs = [
                _build_direct_input(
                    tokenizer=tokenizer,
                    src_text=infer_data[idx]["src_text"],
                    src_lang=infer_data[idx]["src_lang"],
                    trg_lang=infer_data[idx]["trg_lang"],
                    chat_template=chat_template,
                )
                for idx in missing_direct_indices
            ]
            responses, tokens_list = generate_batch(llm, direct_inputs, sampling_params)
            for idx, resp, tok in zip(missing_direct_indices, responses, tokens_list):
                infer_data[idx]["direct_target_text"] = resp
                infer_data[idx]["direct_processed_tokens"] = tok

        # 1b. Populate pivot{i}_text (s -> p_i) if missing, then generate pivot{i}_target_text (p_i -> t)
        for i in range(1, max_pivot + 1):
            lang_key = f"pivot{i}_lang"
            pivot_text_key = f"pivot{i}_text"
            target_key = f"pivot{i}_target_text"

            # Generate missing first-hop source -> pivot translations
            missing_s2p_indices = [
                idx
                for idx, sample in enumerate(infer_data)
                if lang_key in sample and sample.get(pivot_text_key) in (None, "")
            ]
            if missing_s2p_indices:
                print(
                    f"Generating missing {pivot_text_key} (src -> pivot{i}) "
                    f"for {len(missing_s2p_indices)} samples..."
                )
                s2p_inputs = [
                    _build_direct_input(
                        tokenizer=tokenizer,
                        src_text=infer_data[idx]["src_text"],
                        src_lang=infer_data[idx]["src_lang"],
                        trg_lang=infer_data[idx][lang_key],
                        chat_template=chat_template,
                    )
                    for idx in missing_s2p_indices
                ]
                s2p_responses, _ = generate_batch(llm, s2p_inputs, sampling_params)
                for idx, resp in zip(missing_s2p_indices, s2p_responses):
                    infer_data[idx][pivot_text_key] = resp

            # Generate second-hop pivot -> target translations
            valid_indices = []
            p2t_inputs = []
            for idx, sample in enumerate(infer_data):
                if lang_key in sample and sample.get(pivot_text_key) not in (None, ""):
                    valid_indices.append(idx)
                    p2t_inputs.append(
                        _build_direct_input(
                            tokenizer=tokenizer,
                            src_text=sample[pivot_text_key],
                            src_lang=sample[lang_key],
                            trg_lang=sample["trg_lang"],
                            chat_template=chat_template,
                        )
                    )

            if not valid_indices:
                continue

            print(f"Translating {target_key} ({len(valid_indices)} samples)...")
            responses, processed_tokens = generate_batch(
                llm, p2t_inputs, sampling_params
            )
            for idx, response, tokens in zip(
                valid_indices, responses, processed_tokens
            ):
                infer_data[idx][target_key] = response
                infer_data[idx][f"pivot{i}_processed_tokens"] = tokens

    # -------------------------------------------------------
    # Mode 2: Stage 3 - MPP Candidate Aggregation
    # -------------------------------------------------------
    elif target_reference_prompt:
        print("Running MPP Stage 3: Candidate Aggregation (target_reference_prompt)...")

        input_texts = []
        gen_indices = []

        for idx, sample in enumerate(infer_data):
            src_text = sample["src_text"]
            src_lang = sample["src_lang"]
            trg_lang = sample["trg_lang"]

            ref_block = {}
            for i in range(1, max_pivot + 1):
                lang_key = f"pivot{i}_lang"
                text_key = f"pivot{i}_target_text"
                comet_key = f"pivot{i}_comet"
                if lang_key in sample and text_key in sample:
                    pivot_lang = sample[lang_key]
                    pivot_text = sample[text_key]
                    pivot_comet = sample.get(comet_key)
                    if pivot_text and str(pivot_text).strip():
                        ref_block[pivot_lang] = pivot_text
                        ref_block[f"{pivot_lang}_comet"] = pivot_comet

            if ref_block:
                raw_prompt = get_eval_target_reference_prompt(
                    src_tag=src_lang,
                    trg_tag=trg_lang,
                    src_text=src_text,
                    ref_block=ref_block,
                )
                if chat_template:
                    messages = [{"role": "user", "content": raw_prompt}]
                    formatted_prompt = _format_chat_prompt(
                        tokenizer, messages, enable_thinking=False
                    )
                else:
                    formatted_prompt = raw_prompt

                input_texts.append(formatted_prompt)
                gen_indices.append(idx)
            else:
                # Fallback to direct translation when no pivot candidate passes QE filtering
                if reuse_direct_fallback and sample.get("direct_target_text"):
                    sample["mt_text"] = post_clean(sample["direct_target_text"])
                    if save_prompt:
                        sample["prompt"] = _build_direct_input(
                            tokenizer=tokenizer,
                            src_text=src_text,
                            src_lang=src_lang,
                            trg_lang=trg_lang,
                            chat_template=chat_template,
                        )
                else:
                    formatted_prompt = _build_direct_input(
                        tokenizer=tokenizer,
                        src_text=src_text,
                        src_lang=src_lang,
                        trg_lang=trg_lang,
                        chat_template=chat_template,
                    )
                    input_texts.append(formatted_prompt)
                    gen_indices.append(idx)

        if input_texts:
            responses, processed_tokens = generate_batch(
                llm, input_texts, sampling_params
            )
            for idx, response, prompt, tokens in zip(
                gen_indices, responses, input_texts, processed_tokens
            ):
                infer_data[idx]["mt_text"] = response
                infer_data[idx]["processed_tokens"] = tokens
                if save_prompt:
                    infer_data[idx]["prompt"] = prompt

    # -------------------------------------------------------
    # Mode 3: Direct Translation
    # -------------------------------------------------------
    else:
        print("Running Direct Translation mode...")
        src_list = [sample["src_text"] for sample in infer_data]
        src_langs = [sample["src_lang"] for sample in infer_data]
        trg_langs = [sample["trg_lang"] for sample in infer_data]

        out_dict = func_call(
            model=llm,
            tokenizer=tokenizer,
            src_list=src_list,
            src_langs=src_langs,
            trg_langs=trg_langs,
            chat_template=chat_template,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            temperature=temperature,
        )

        for sample, response, prompt, tokens in zip(
            infer_data,
            out_dict["response"],
            out_dict["prompt"],
            out_dict["processed_tokens"],
        ):
            sample["mt_text"] = response
            sample["processed_tokens"] = tokens
            if save_prompt:
                sample["prompt"] = prompt

    if out_data_path is None:
        out_data_path = infer_data_path

    out_file = Path(out_data_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(infer_data, f, ensure_ascii=False, indent=2)

    print(f"Saved inference outputs ({len(infer_data)} samples) to {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
