import logging
import os
import gc
from copy import deepcopy
from statistics import mean
from typing import Dict, Any

import optuna
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

from model import get_classifier

logger = logging.getLogger(__name__)


# ===============================
# ヘルパー：YAMLにキーがあるときだけ上書き（struct安全）
# ===============================
def _set_if_exists(cfg, key, value):
    try:
        if key in cfg:
            cfg[key] = value
    except Exception:
        # OmegaConf以外のケースやネストで失敗した場合はスルー
        pass


# ===============================
# パラメータ空間（F1最適化向け）
#  ※ 上限を現実的にしてOOMを抑える
# ===============================


def xgboost_config(trial: optuna.Trial, model_config, name: str = ""):
    model_config.max_depth = trial.suggest_int("max_depth", 4, 10)
    model_config.learning_rate = trial.suggest_float(
        "learning_rate", 1e-3, 0.2, log=True
    )
    model_config.n_estimators = trial.suggest_int("n_estimators", 200, 1500)

    model_config.min_child_weight = trial.suggest_float(
        "min_child_weight", 1e-2, 10.0, log=True
    )
    model_config.gamma = trial.suggest_float("gamma", 0.0, 5.0)
    model_config.reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True)
    model_config.reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True)

    model_config.subsample = trial.suggest_float("subsample", 0.5, 0.9)
    model_config.colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.8)

    if int(model_config.max_depth) >= 10:
        model_config.learning_rate = min(float(model_config.learning_rate), 0.1)
    return model_config


def lightgbm_config(trial: optuna.Trial, model_config, name: str = ""):
    model_config.max_depth = trial.suggest_int("max_depth", -1, 10)
    model_config.learning_rate = trial.suggest_float(
        "learning_rate", 1e-3, 0.2, log=True
    )
    model_config.n_estimators = trial.suggest_int("n_estimators", 200, 2000)
    model_config.num_leaves = trial.suggest_int("num_leaves", 16, 256, log=True)

    model_config.reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True)
    model_config.reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True)

    model_config.subsample = trial.suggest_float(
        "subsample", 0.6, 0.9
    )  # bagging_fraction
    model_config.colsample_bytree = trial.suggest_float(
        "colsample_bytree", 0.6, 0.9
    )  # feature_fraction
    model_config.subsample_freq = trial.suggest_int("subsample_freq", 0, 7)
    model_config.min_child_samples = trial.suggest_int(
        "min_child_samples", 10, 150, log=True
    )

    if model_config.max_depth not in (-1, 0):
        max_leaves = 2 ** int(model_config.max_depth)
        if int(model_config.num_leaves) > max_leaves:
            model_config.num_leaves = max_leaves

    # 省メモリ固定（YAMLにあれば）
    _set_if_exists(model_config, "force_row_wise", True)
    _set_if_exists(model_config, "max_bin", 255)
    _set_if_exists(model_config, "n_jobs", 1)
    _set_if_exists(model_config, "early_stopping_rounds", 100)
    _set_if_exists(model_config, "verbosity", -1)
    return model_config


def catboost_config(trial: optuna.Trial, model_config, name: str = ""):
    model_config.depth = trial.suggest_int("depth", 4, 10)
    model_config.learning_rate = trial.suggest_float(
        "learning_rate", 1e-3, 0.2, log=True
    )
    model_config.n_estimators = trial.suggest_int("n_estimators", 200, 4000)
    model_config.l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True)
    model_config.random_strength = trial.suggest_float("random_strength", 0.0, 20.0)
    model_config.subsample = trial.suggest_float("subsample", 0.5, 1.0)
    _set_if_exists(model_config, "early_stopping_rounds", 100)
    return model_config


def xgblgbm_config(trial: optuna.Trial, model_config, name: str = ""):
    # XGBoost
    model_config.xgboost.max_depth = trial.suggest_int("xgboost_max_depth", 4, 10)
    model_config.xgboost.learning_rate = trial.suggest_float(
        "xgboost_learning_rate", 1e-3, 0.2, log=True
    )
    model_config.xgboost.n_estimators = trial.suggest_int(
        "xgboost_n_estimators", 200, 1500
    )
    model_config.xgboost.min_child_weight = trial.suggest_float(
        "xgboost_min_child_weight", 1e-2, 10.0, log=True
    )
    model_config.xgboost.gamma = trial.suggest_float("xgboost_gamma", 0.0, 5.0)
    model_config.xgboost.reg_alpha = trial.suggest_float(
        "xgboost_reg_alpha", 1e-8, 5.0, log=True
    )
    model_config.xgboost.reg_lambda = trial.suggest_float(
        "xgboost_reg_lambda", 1e-8, 5.0, log=True
    )
    model_config.xgboost.subsample = trial.suggest_float("xgboost_subsample", 0.5, 0.9)
    model_config.xgboost.colsample_bytree = trial.suggest_float(
        "xgboost_colsample_bytree", 0.5, 0.8
    )

    _set_if_exists(model_config.xgboost, "tree_method", "hist")
    _set_if_exists(model_config.xgboost, "grow_policy", "lossguide")
    _set_if_exists(model_config.xgboost, "max_bin", 256)
    _set_if_exists(model_config.xgboost, "nthread", 1)
    _set_if_exists(model_config.xgboost, "early_stopping_rounds", 100)
    _set_if_exists(model_config.xgboost, "verbosity", 0)

    # LightGBM
    model_config.lightgbm.max_depth = trial.suggest_int("lightgbm_max_depth", -1, 10)
    model_config.lightgbm.learning_rate = trial.suggest_float(
        "lightgbm_learning_rate", 1e-3, 0.2, log=True
    )
    model_config.lightgbm.n_estimators = trial.suggest_int(
        "lightgbm_n_estimators", 200, 2000
    )
    model_config.lightgbm.num_leaves = trial.suggest_int(
        "lightgbm_num_leaves", 16, 256, log=True
    )
    model_config.lightgbm.reg_alpha = trial.suggest_float(
        "lightgbm_reg_alpha", 1e-8, 5.0, log=True
    )
    model_config.lightgbm.reg_lambda = trial.suggest_float(
        "lightgbm_reg_lambda", 1e-8, 5.0, log=True
    )
    model_config.lightgbm.subsample = trial.suggest_float(
        "lightgbm_subsample", 0.6, 0.9
    )
    model_config.lightgbm.colsample_bytree = trial.suggest_float(
        "lightgbm_colsample_bytree", 0.6, 0.9
    )
    model_config.lightgbm.subsample_freq = trial.suggest_int(
        "lightgbm_subsample_freq", 0, 7
    )
    model_config.lightgbm.min_child_samples = trial.suggest_int(
        "lightgbm_min_child_samples", 10, 150, log=True
    )

    if model_config.lightgbm.max_depth not in (-1, 0):
        max_leaves = 2 ** int(model_config.lightgbm.max_depth)
        if int(model_config.lightgbm.num_leaves) > max_leaves:
            model_config.lightgbm.num_leaves = max_leaves

    _set_if_exists(model_config.lightgbm, "force_row_wise", True)
    _set_if_exists(model_config.lightgbm, "max_bin", 255)
    _set_if_exists(model_config.lightgbm, "n_jobs", 1)
    _set_if_exists(model_config.lightgbm, "early_stopping_rounds", 100)
    _set_if_exists(model_config.lightgbm, "verbosity", -1)

    return model_config


def get_model_config(model_name: str):
    if model_name == "xgboost":
        return xgboost_config
    elif model_name == "lightgbm":
        return lightgbm_config
    elif model_name == "catboost":
        return catboost_config
    elif model_name == "xgblgbm":
        return xgblgbm_config
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def update_model_config(default_config: Dict[str, Any], best_config: Dict[str, Any]):
    for _p, v in best_config.items():
        if _p.startswith("xgboost_"):
            model_name = "xgboost"
            param_name = _p[len("xgboost_") :]
        elif _p.startswith("lightgbm_"):
            model_name = "lightgbm"
            param_name = _p[len("lightgbm_") :]
        else:
            model_name = None
            param_name = _p

        if model_name is None:
            if param_name in default_config:
                default_config[param_name] = v
        elif model_name in default_config:
            default_config[model_name][param_name] = v
    return default_config


# ===============================
# 最適化本体（サンプリング + 早期終了 + 省メモリ）
# ===============================


class OptimParam:
    def __init__(
        self,
        model_name,
        default_config,
        input_dim,
        output_dim,
        X,
        y,
        val_data,
        columns,
        target_column,
        n_trials,
        n_startup_trials,
        storage,
        study_name,
        cv=False,  # True / False / int (fold数)
        n_jobs=1,
        seed=42,
        alpha=1,
        task="None",
        threshold: float = 0.5,
        # ← 追加：検索時のみのサンプリング率（OOM回避）
        row_frac_for_search: float = 0.6,
        col_frac_for_search: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.default_config = deepcopy(default_config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = get_model_config(model_name)
        self.X = X
        self.y = y
        self.val_data = val_data
        self.columns = columns
        self.target_column = target_column
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.storage = to_absolute_path(storage) if storage is not None else None
        self.study_name = study_name
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self.alpha = alpha
        self.task = task
        self.threshold = float(threshold)
        self.row_frac_for_search = float(row_frac_for_search)
        self.col_frac_for_search = float(col_frac_for_search)

        if isinstance(cv, bool):
            self.cv_n_splits = 5 if cv else 0
        elif isinstance(cv, int):
            self.cv_n_splits = max(2, cv)
        else:
            self.cv_n_splits = 0

    def _maybe_sample(self, X_train, y_train, X_val):
        """検索時のみ行・列サンプリング（CV/oneshot共通）。"""
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        y_train = pd.Series(np.asarray(y_train), index=X_train.index)

        # 行サンプル
        if 0 < self.row_frac_for_search < 1.0:
            X_train = X_train.sample(
                frac=self.row_frac_for_search, random_state=self.seed
            )
            y_train = y_train.loc[X_train.index]

        # 列サンプル（検証側も同列に合わせる）
        if 0 < self.col_frac_for_search < 1.0:
            n_cols = X_train.shape[1]
            rng = np.random.RandomState(self.seed)
            sel = rng.choice(
                X_train.columns,
                size=max(1, int(n_cols * self.col_frac_for_search)),
                replace=False,
            )
            X_train = X_train[sel]
            if X_val is not None:
                if not isinstance(X_val, pd.DataFrame):
                    X_val = pd.DataFrame(X_val, columns=self.columns)
                X_val = X_val[sel]
        return X_train, y_train, X_val

    def fit(
        self, model_config, X_train, y_train, X_val=None, y_val=None
    ) -> Dict[str, float]:
        # 検証データセット（デフォルトは self.val_data）
        if X_val is None or y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()

        # 検索時はサンプリング（OOM対策）
        X_train, y_train, X_val = self._maybe_sample(X_train, y_train, X_val)

        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            seed=self.seed,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        score = (
            model.evaluate(X_val, y_val, threshold=self.threshold)
            if hasattr(model, "evaluate")
            else {"F1": 0.0}
        )

        # 後片付け
        del model
        gc.collect()
        return score

    def cross_validation(self, model_config) -> float:
        skf = StratifiedKFold(
            n_splits=self.cv_n_splits, random_state=self.seed, shuffle=True
        )
        f1_list = []
        for _, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, y_train = self.X.iloc[train_idx], self.y[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]
            score = self.fit(model_config, X_train, y_train, X_val, y_val)
            f1_list.append(float(score.get("F1", 0.0)))
        return mean(f1_list) if f1_list else 0.0

    def one_shot(self, model_config) -> float:
        X_val = self.val_data[self.columns]
        y_val = self.val_data[self.target_column].values.squeeze()
        score = self.fit(model_config, self.X, self.y, X_val, y_val)
        return float(score.get("F1", 0.0))

    def objective(self, trial: optuna.Trial) -> float:
        _model_config = self.model_config(trial, deepcopy(self.default_config))
        if self.cv_n_splits >= 2:
            value = self.cross_validation(_model_config)
        else:
            value = self.one_shot(_model_config)
        trial.set_user_attr("f1", value)
        return value

    @staticmethod
    def _n_complete(study: optuna.Study) -> int:
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    def get_best_config(self) -> Dict[str, Any]:
        storage_obj = None
        if self.storage is not None:
            os.makedirs(self.storage, exist_ok=True)
            storage_obj = optuna.storages.RDBStorage(
                url=f"sqlite:///{self.storage}/optuna.db"
            )

        study = optuna.create_study(
            storage=storage_obj,
            study_name=self.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self.seed, n_startup_trials=self.n_startup_trials
            ),
            load_if_exists=True,
        )

        n_complete = self._n_complete(study)
        n_trials = max(0, int(self.n_trials) - n_complete)
        if n_trials == 0 and n_complete > 0:
            logger.info(
                f"[Optuna] Study already has {n_complete} complete trials. No new trials scheduled."
            )
        else:
            logger.info(
                f"[Optuna] Running {n_trials} trial(s) (already complete: {n_complete})."
            )

        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)

        logger.info(f"[Optuna] Best F1={study.best_value:.6f}")
        logger.info(f"[Optuna] Best params={study.best_params}")
        update_model_config(self.default_config, study.best_params)
        return self.default_config
