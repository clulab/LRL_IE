def strip_think(text: str) -> str:
    """
    If the model emits a reasoning block, keep only the content after the closing </think> tag.
    If no </think> tag is present, return the text unchanged (trimmed).
    """
    if "</think>" not in text:
        return text.strip()
    return text.rsplit("</think>", 1)[1].strip()


def chatml_prompt(system_text: str, user_text: str, no_think: bool = True) -> str:
    """
    Wrap system + user turns into a minimal ChatML prompt for Qwen-style models.
    """
    if no_think:
        system_text = "/no_think\n" + system_text
    return (
        "<|im_start|>system\n" + system_text.strip() + "\n<|im_end|>\n"
        "<|im_start|>user\n" + user_text.strip() + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
