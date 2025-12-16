from typing import Dict

from .io import load_yaml


def load_experiment_config(base_path: str, experiments_path: str, experiment_name: str, model_name: str) -> Dict:
    base_cfg = load_yaml(base_path) or {}
    model_cfg = base_cfg.get(model_name)
    if model_cfg is None:
        raise KeyError(f"Model '{model_name}' not found in {base_path}")

    exp_cfgs = load_yaml(experiments_path) or {}
    exp_cfg = exp_cfgs.get(experiment_name, {})

    cfg = dict(model_cfg)
    cfg.update(exp_cfg)
    cfg["model"] = model_name
    return cfg
