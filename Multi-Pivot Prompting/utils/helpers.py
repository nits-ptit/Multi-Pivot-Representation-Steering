import re
import warnings


def post_clean(text: str) -> str:
    """Clean LLM-generated translation output."""
    if text is None:
        return ""

    text = text.strip()

    # Keep only the content after </think> for reasoning models (e.g., Qwen3.5)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    # Remove any remaining <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove any leftover think tags
    text = text.replace("<think>", "").replace("</think>", "")

    # Replace newlines with spaces/semicolons into a single line
    if "\n" in text:
        warnings.warn("Output contains multiple lines; concatenating into a single line.")
        text = text.replace("\n", "; ")

    # Remove repeated semicolons and surrounding spaces
    text = re.sub(r"\s*;\s*", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def flat_text(text: str) -> str:
    """Flatten newlines and strip whitespace for metric evaluation."""
    if text is None:
        return ""
    return text.replace("\n", " ").strip()
