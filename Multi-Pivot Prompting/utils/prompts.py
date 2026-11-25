from utils.configs import lang_tag_map

DIRECT_SYSTEM_PROMPT = (
    "You are a professional translator. Output ONLY the translation. "
    "Do not explain. Do not add notes. Do not add extra text."
)


def get_eval_prompt_direct_og(src_tag: str, trg_tag: str) -> str:
    """
    Direct Translation Prompt (Appendix A.1).
    Uses '{}' placeholder for the source sentence.
    """
    src_lang = lang_tag_map.get(src_tag, src_tag)
    trg_lang = lang_tag_map.get(trg_tag, trg_tag)
    return (
        f"Translate the following text from {src_lang} into {trg_lang}:\n"
        f"{src_lang}: {{}}\n"
        f"{trg_lang}:"
    )


def get_eval_target_reference_prompt(
    src_tag: str,
    trg_tag: str,
    src_text: str,
    ref_block: dict,
) -> str:
    """
    Multi-Pivot Prompting (MPP) Candidate Aggregation Template (Appendix A.2).

    Args:
        src_tag: Source ISO-2 language code (e.g., "en").
        trg_tag: Target ISO-2 language code (e.g., "fr").
        src_text: Original source sentence.
        ref_block: Dict mapping `{pivot_lang: pivot_target_text, f"{pivot_lang}_comet": qe_score}`.
    """
    src_lang = lang_tag_map.get(src_tag, src_tag)
    trg_lang = lang_tag_map.get(trg_tag, trg_tag)

    references = []
    for key, value in ref_block.items():
        if not key.endswith("_comet") and key != "direct_comet":
            text = value
            comet = ref_block.get(f"{key}_comet", None)

            if text and str(text).strip():
                if comet is not None:
                    references.append((str(text).strip(), float(comet)))
                else:
                    references.append((str(text).strip(), None))

    # Sort references by quality score (high -> low)
    references = sorted(
        references,
        key=lambda x: x[1] if x[1] is not None else 0.0,
        reverse=True,
    )

    formatted_refs = "\n".join(
        f"- {text} (quality: {score:.3f})" if score is not None else f"- {text}"
        for text, score in references
    )

    prompt = (
        f"You are an expert multilingual translator specializing in {src_lang} → {trg_lang}.\n\n"
        f"## Task\n"
        f"Translate the source sentence into {trg_lang}. "
        f"Reference translations from intermediate pivot systems are provided to assist you, "
        f"but they may contain errors — treat them as imperfect hints, not ground truth.\n\n"
        f"## Source ({src_lang})\n"
        f"{src_text}\n\n"
        f"## Pivot Reference Translations\n"
        f"Each reference has a quality score (0–1, higher = better).\n"
        f"{formatted_refs}\n\n"
        f"## How to use the references\n"
        f"1. **Source is authoritative.** If a reference contradicts the source "
        f"(wrong names, wrong facts, omissions), trust the source.\n"
        f"2. **Weight by quality score.** Higher-scored references are more reliable "
        f"for phrasing and fluency — prefer them, but verify against the source.\n"
        f"3. **Resolve conflicts critically.** When references disagree, check the source "
        f"to decide which is correct — do not average or blend conflicting facts.\n"
        f"4. **Named entities and numbers.** Copy proper nouns, names, and numbers "
        f"directly from the source if references render them inconsistently.\n"
        f"5. **Improve, don't just copy.** You may borrow fluent phrasing from references, "
        f"but your final translation must be factually accurate and naturally idiomatic.\n\n"
        f"## Output constraints\n"
        f"- Output ONLY the final {trg_lang} translation.\n"
        f"- No notes, no explanations, no alternatives.\n\n"
        f"{trg_lang}:"
    )

    return prompt
