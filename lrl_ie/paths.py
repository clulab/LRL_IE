from pathlib import Path


def resolve_templated_path(template: str, experiment: str, model: str) -> Path:
    rendered = template.replace("{experiment}", experiment).replace("{model}", model)
    return Path(rendered)


def resolve_pred_out_path(out_template: str, experiment: str, model: str) -> Path:
    out_path = resolve_templated_path(out_template, experiment, model)
    if model not in out_path.stem:
        out_path = out_path.with_name(f"{out_path.stem}_{model}{out_path.suffix}")
    return out_path

