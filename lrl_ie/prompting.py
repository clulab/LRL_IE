from typing import Dict, List, Tuple

from llama_cpp import Llama


def count_tokens(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def render_user_template(user_template: str, input_text: str) -> str:
    return user_template.replace("{input_text}", input_text)


def fit_system_with_examples(llm: Llama, ctx: int, max_gen_tokens: int, system_prompt_base: str, fewshots: List[str]) -> str:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    tokens = count_tokens(llm, system_prompt_base)
    pieces: List[str] = []
    for ex in fewshots:
        ex_tokens = count_tokens(llm, ex)
        if tokens + ex_tokens > budget:
            break
        pieces.append(ex)
        tokens += ex_tokens
    return system_prompt_base + "".join(pieces)


def maybe_truncate_input(llm: Llama, input_text: str, ctx: int, sys_str: str, max_gen_tokens: int, user_template: str) -> Tuple[str, int, int]:
    SAFETY = 64
    budget = ctx - max_gen_tokens - SAFETY
    user_block = render_user_template(user_template, input_text=input_text)
    total = count_tokens(llm, sys_str) + count_tokens(llm, user_block)
    if total <= budget:
        return input_text, total, budget
    print(f"truncating input: {total} > {budget}")
    toks_input = llm.tokenize(input_text.encode("utf-8"), add_bos=False)
    keep = max(64, budget - count_tokens(llm, sys_str) - count_tokens(llm, render_user_template(user_template, input_text="")))
    truncated_ids = toks_input[-keep:]
    truncated_text = llm.detokenize(truncated_ids).decode("utf-8", errors="ignore")
    user_block2 = render_user_template(user_template, input_text=truncated_text)
    total2 = count_tokens(llm, sys_str) + count_tokens(llm, user_block2)
    return truncated_text, total2, budget


def build_messages(llm: Llama, input_text: str, ctx: int, max_gen_tokens: int, prompt_data: dict, fewshots: List[str]) -> List[Dict[str, str]]:
    system_str = fit_system_with_examples(llm, ctx, max_gen_tokens, prompt_data["prompt_base"], fewshots)
    input_text_fitted, _, _ = maybe_truncate_input(llm, input_text, ctx, system_str, max_gen_tokens, prompt_data["user_template"])
    return [
        {"role": "system", "content": system_str},
        {"role": "user", "content": render_user_template(prompt_data["user_template"], input_text=input_text_fitted.strip())},
    ]
