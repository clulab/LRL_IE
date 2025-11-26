# utils.py

import json
import re

def load_dataset(path):
    """Load JSON dataset from the given file path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_prompt(template_path, sentence):
    """Read a prompt template and substitute {sentence}."""
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    return template.replace("{sentence}", sentence)

def parse_json_output(text):
    """Extract JSON object from a longer LLM output string."""
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return json.loads(text[start:end])
    except Exception as e:
        return {"error": f"Parsing failed: {e}"}
    
def apply_prompt_template(template: str, tokens: list) -> str:
    """
    Fills the template with the given list of tokens.
    """
    sentence = " ".join(tokens)
    return template.replace("{sentence}", sentence.strip())
