from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.modeling.builders import (
    build_is_win_trainer,
    is_win_bundle,
)
from src.modeling.tuning import tune_is_win
from src.modeling.trainer import ModelTrainer

BEST_PARAMS_PATH = Path("artifacts/is_win_best_params.json")


def load_best_params() -> Dict[str, Dict]:
    if BEST_PARAMS_PATH.exists():
        return json.loads(BEST_PARAMS_PATH.read_text())
    return {}


def save_best_params(data: Dict[str, Dict]) -> None:
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def update_best_params(results) -> None:
    cache = load_best_params()
    for res in results:
        cache[res.model] = {"params": res.best_params, "metric": res.best_value}
    save_best_params(cache)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/tune IS_WIN models using Feast features.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before training.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials per model.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of models to train/tune (e.g. lgbm xgb).",
    )
    parser.add_argument("--skip-training", action="store_true", help="Skip final training phase.")
    parser.add_argument("--metric", default="log_loss", help="Metric used during tuning (log_loss, roc_auc...).")
    return parser.parse_args()


def _normalize_stacking_params(params: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not params:
        return params
    normalized = {}
    for key, value in params.items():
        if key in {"stack_alpha", "final_estimator__alpha"}:
            normalized["final_estimator__alpha"] = value
        elif key in {"stack_C", "final_estimator__C"}:
            normalized["final_estimator__C"] = value
        else:
            normalized[key] = value
    return normalized


def train_with_best_params(trainer: ModelTrainer, models: Optional[List[str]] = None) -> pd.DataFrame:
    cache = load_best_params()
    rows = []
    target_models = models or trainer.available_models()
    for model_key in target_models:
        overrides = cache.get(model_key, {}).get("params")
        if overrides and model_key == "stacking":
            overrides = _normalize_stacking_params(overrides)
        metrics = trainer.train_with_params(
            model_key,
            param_overrides=overrides,
            run_name=f"{model_key}_tuned" if overrides else model_key,
        )
        rows.append({"model": model_key, "tuned": overrides is not None, **metrics})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    bundle = is_win_bundle()
    trainer = build_is_win_trainer()

    if args.tune:
        results = tune_is_win(
            bundle.dataset,
            bundle.training,
            models=args.models,
            n_trials=args.trials,
            metric=args.metric,
        )
        update_best_params(results)
        for res in results:
            print(f"Tuned {res.model}: {res.best_value:.4f} -> {res.best_params}")

    if not args.skip_training:
        df = train_with_best_params(trainer, models=args.models)
        print(df)


if __name__ == "__main__":
    main()
