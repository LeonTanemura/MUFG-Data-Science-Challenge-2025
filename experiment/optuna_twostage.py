import logging
import os
import gc
from copy import deepcopy
from statistics import mean
from typing import Dict, Any, Tuple, Optional

import optuna
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

from model import get_classifier

logger = logging.getLogger(__name__)


# =============================================================================
# 共通: パラメータサジェスト用ユーティリティ（縮小空間に対応）
# =============================================================================


def _suggest_int(
    trial: optuna.Trial,
    name: str,
    low: int,
    high: int,
    *,
    log: bool = False,
    overrides: Optional[Dict[str, Tuple[float, float, bool]]] = None,
) -> int:
    if overrides and name in overrides:
        low, high, log = overrides[name]
    return trial.suggest_int(name, int(low), int(high), log=log)


def _suggest_float(
    trial: optuna.Trial,
    name: str,
    low: float,
    high: float,
    *,
    log: bool = False,
    overrides: Optional[Dict[str, Tuple[float, float, bool]]] = None,
) -> float:
    if overrides and name in overrides:
        low, high, log = overrides[name]
    return trial.suggest_float(name, float(low), float(high), log=log)


# =============================================================================
# 検索空間（F1 最適化を想定）。Stage1(light=True) は軽量レンジに。
# ※ ここで触るキーは YAML に全て定義されている前提です（struct error 回避）。
# =============================================================================


def xgboost_config(
    trial: optuna.Trial,
    model_config,
    name: str = "",
    overrides: Optional[Dict[str, Tuple[float, float, bool]]] = None,
    *,
    light: bool = False,
):
    if light:
        n_estimators_hi = 1200
        max_depth_hi = 8
        lr_hi = 0.2
        col_hi = 0.8
        sub_hi = 0.9
        gamma_hi = 5.0
        mcw_hi = 10.0
        reg_hi = 5.0
    else:
        n_estimators_hi = 3000
        max_depth_hi = 12
        lr_hi = 0.3
        col_hi = 1.0
        sub_hi = 1.0
        gamma_hi = 10.0
        mcw_hi = 20.0
        reg_hi = 10.0

    model_config.max_depth = _suggest_int(
        trial, "max_depth", 4, max_depth_hi, overrides=overrides
    )
    model_config.learning_rate = _suggest_float(
        trial, "learning_rate", 1e-3, lr_hi, log=True, overrides=overrides
    )
    model_config.n_estimators = _suggest_int(
        trial, "n_estimators", 200, n_estimators_hi, overrides=overrides
    )

    model_config.min_child_weight = _suggest_float(
        trial, "min_child_weight", 1e-2, mcw_hi, log=True, overrides=overrides
    )
    model_config.gamma = _suggest_float(
        trial, "gamma", 0.0, gamma_hi, overrides=overrides
    )
    model_config.reg_alpha = _suggest_float(
        trial, "reg_alpha", 1e-8, reg_hi, log=True, overrides=overrides
    )
    model_config.reg_lambda = _suggest_float(
        trial, "reg_lambda", 1e-8, reg_hi, log=True, overrides=overrides
    )

    model_config.subsample = _suggest_float(
        trial, "subsample", 0.5, sub_hi, overrides=overrides
    )
    model_config.colsample_bytree = _suggest_float(
        trial, "colsample_bytree", 0.5, col_hi, overrides=overrides
    )

    # 深い木なら学習率を抑制
    if int(model_config.max_depth) >= 10:
        model_config.learning_rate = min(float(model_config.learning_rate), 0.1)

    return model_config


def lightgbm_config(
    trial: optuna.Trial,
    model_config,
    name: str = "",
    overrides: Optional[Dict[str, Tuple[float, float, bool]]] = None,
    *,
    light: bool = False,
):
    if light:
        n_estimators_hi = 1200
        max_depth_hi = 8
        lr_hi = 0.2
        col_hi = 0.9
        sub_hi = 0.9
        min_child_hi = 100
        num_leaves_hi = 256
    else:
        n_estimators_hi = 3000
        max_depth_hi = 12
        lr_hi = 0.3
        col_hi = 1.0
        sub_hi = 1.0
        min_child_hi = 200
        num_leaves_hi = 512

    model_config.max_depth = _suggest_int(
        trial, "max_depth", -1, max_depth_hi, overrides=overrides
    )
    model_config.learning_rate = _suggest_float(
        trial, "learning_rate", 1e-3, lr_hi, log=True, overrides=overrides
    )
    model_config.n_estimators = _suggest_int(
        trial, "n_estimators", 200, n_estimators_hi, overrides=overrides
    )
    model_config.num_leaves = _suggest_int(
        trial, "num_leaves", 16, num_leaves_hi, log=True, overrides=overrides
    )

    model_config.reg_alpha = _suggest_float(
        trial, "reg_alpha", 1e-8, 10.0, log=True, overrides=overrides
    )
    model_config.reg_lambda = _suggest_float(
        trial, "reg_lambda", 1e-8, 10.0, log=True, overrides=overrides
    )

    model_config.subsample = _suggest_float(
        trial, "subsample", 0.6, sub_hi, overrides=overrides
    )  # bagging_fraction
    model_config.colsample_bytree = _suggest_float(
        trial, "colsample_bytree", 0.6, col_hi, overrides=overrides
    )  # feature_fraction
    model_config.subsample_freq = _suggest_int(
        trial, "subsample_freq", 0, 7, overrides=overrides
    )
    model_config.min_child_samples = _suggest_int(
        trial, "min_child_samples", 5, min_child_hi, log=True, overrides=overrides
    )

    # num_leaves と max_depth の緩い整合
    if model_config.max_depth not in (-1, 0):
        max_leaves = 2 ** int(model_config.max_depth)
        if int(model_config.num_leaves) > max_leaves:
            model_config.num_leaves = max_leaves

    return model_config


def catboost_config(
    trial: optuna.Trial,
    model_config,
    name: str = "",
    overrides: Optional[Dict[str, Tuple[float, float, bool]]] = None,
    *,
    light: bool = False,
):
    if light:
        n_estimators_hi = 2000
        lr_hi = 0.2
    else:
        n_estimators_hi = 6000
        lr_hi = 0.3

    model_config.depth = _suggest_int(trial, "depth", 4, 10, overrides=overrides)
    model_config.learning_rate = _suggest_float(
        trial, "learning_rate", 1e-3, lr_hi, log=True, overrides=overrides
    )
    model_config.n_estimators = _suggest_int(
        trial, "n_estimators", 200, n_estimators_hi, overrides=overrides
    )
    model_config.l2_leaf_reg = _suggest_float(
        trial, "l2_leaf_reg", 1.0, 20.0, log=True, overrides=overrides
    )
    model_config.random_strength = _suggest_float(
        trial, "random_strength", 0.0, 20.0, overrides=overrides
    )
    model_config.subsample = _suggest_float(
        trial, "subsample", 0.5, 1.0, overrides=overrides
    )
    return model_config


def xgblgbm_config(
    trial: optuna.Trial,
    model_config,
    name: str = "",
    overrides: Optional[Dict[str, Tuple[float, float, bool]]] = None,
    *,
    light: bool = False,
):
    # ---- XGBoost ----
    model_config.xgboost.max_depth = _suggest_int(
        trial, "xgboost_max_depth", 4, (8 if light else 12), overrides=overrides
    )
    model_config.xgboost.learning_rate = _suggest_float(
        trial,
        "xgboost_learning_rate",
        1e-3,
        (0.2 if light else 0.3),
        log=True,
        overrides=overrides,
    )
    model_config.xgboost.n_estimators = _suggest_int(
        trial,
        "xgboost_n_estimators",
        200,
        (1200 if light else 3000),
        overrides=overrides,
    )
    model_config.xgboost.min_child_weight = _suggest_float(
        trial,
        "xgboost_min_child_weight",
        1e-2,
        (10.0 if light else 20.0),
        log=True,
        overrides=overrides,
    )
    model_config.xgboost.gamma = _suggest_float(
        trial, "xgboost_gamma", 0.0, (5.0 if light else 10.0), overrides=overrides
    )
    model_config.xgboost.reg_alpha = _suggest_float(
        trial,
        "xgboost_reg_alpha",
        1e-8,
        (5.0 if light else 10.0),
        log=True,
        overrides=overrides,
    )
    model_config.xgboost.reg_lambda = _suggest_float(
        trial,
        "xgboost_reg_lambda",
        1e-8,
        (5.0 if light else 10.0),
        log=True,
        overrides=overrides,
    )
    model_config.xgboost.subsample = _suggest_float(
        trial, "xgboost_subsample", 0.5, (0.9 if light else 1.0), overrides=overrides
    )
    model_config.xgboost.colsample_bytree = _suggest_float(
        trial,
        "xgboost_colsample_bytree",
        0.5,
        (0.8 if light else 1.0),
        overrides=overrides,
    )

    # ---- LightGBM ----
    model_config.lightgbm.max_depth = _suggest_int(
        trial, "lightgbm_max_depth", -1, (8 if light else 12), overrides=overrides
    )
    model_config.lightgbm.learning_rate = _suggest_float(
        trial,
        "lightgbm_learning_rate",
        1e-3,
        (0.2 if light else 0.3),
        log=True,
        overrides=overrides,
    )
    model_config.lightgbm.n_estimators = _suggest_int(
        trial,
        "lightgbm_n_estimators",
        200,
        (1200 if light else 3000),
        overrides=overrides,
    )
    model_config.lightgbm.num_leaves = _suggest_int(
        trial,
        "lightgbm_num_leaves",
        16,
        (256 if light else 512),
        log=True,
        overrides=overrides,
    )
    model_config.lightgbm.reg_alpha = _suggest_float(
        trial, "lightgbm_reg_alpha", 1e-8, 10.0, log=True, overrides=overrides
    )
    model_config.lightgbm.reg_lambda = _suggest_float(
        trial, "lightgbm_reg_lambda", 1e-8, 10.0, log=True, overrides=overrides
    )
    model_config.lightgbm.subsample = _suggest_float(
        trial, "lightgbm_subsample", 0.6, (0.9 if light else 1.0), overrides=overrides
    )
    model_config.lightgbm.colsample_bytree = _suggest_float(
        trial,
        "lightgbm_colsample_bytree",
        0.6,
        (0.9 if light else 1.0),
        overrides=overrides,
    )
    model_config.lightgbm.subsample_freq = _suggest_int(
        trial, "lightgbm_subsample_freq", 0, 7, overrides=overrides
    )
    model_config.lightgbm.min_child_samples = _suggest_int(
        trial,
        "lightgbm_min_child_samples",
        5,
        (100 if light else 200),
        log=True,
        overrides=overrides,
    )

    if model_config.lightgbm.max_depth not in (-1, 0):
        max_leaves = 2 ** int(model_config.lightgbm.max_depth)
        if int(model_config.lightgbm.num_leaves) > max_leaves:
            model_config.lightgbm.num_leaves = max_leaves

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
    """
    study.best_params を default_config に反映。
    - xgblgbm のようなネスト（'xgboost_' / 'lightgbm_' プレフィックス）にも対応。
    - それ以外はトップレベルへ上書き。
    """
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


# =============================================================================
# ベース最適化クラス（単段）
# =============================================================================


class OptimParam:
    """
    既存互換の単段最適化（F1 を最大化）。
    """

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
        cv=False,  # True/False または int（fold数）
        n_jobs=1,
        seed=42,
        alpha=1,
        task="None",
        threshold: float = 0.5,
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
        self.n_trials = int(n_trials)
        self.n_startup_trials = int(n_startup_trials)
        self.storage = to_absolute_path(storage) if storage is not None else None
        self.study_name = study_name
        self.n_jobs = n_jobs
        self.seed = seed
        self.alpha = alpha
        self.task = task
        self.threshold = float(threshold)

        if isinstance(cv, bool):
            self.cv_n_splits = 5 if cv else 0
        elif isinstance(cv, int):
            self.cv_n_splits = max(2, cv)
        else:
            self.cv_n_splits = 0

        # 二段探索で使う
        self.overrides: Optional[Dict[str, Tuple[float, float, bool]]] = None
        self._light_space: bool = False
        self._stage1_mode: bool = False
        self.stage1_row_frac: float = 0.5  # 行サンプリング率
        self.stage1_col_frac: float = 0.6  # 列サンプリング率

    def fit(
        self, model_config, X_train, y_train, X_val=None, y_val=None
    ) -> Dict[str, float]:
        import numpy as np
        import pandas as pd

        # ====== 1) y を必ず X の index に揃える ======
        #  - CV でも one-shot でも、ここで y を Series(index=X_train.index) に統一
        if isinstance(X_train, pd.DataFrame):
            # y_train が ndarray/リスト/Series いずれでも、X_train と同じ index を付与
            y_train = pd.Series(np.asarray(y_train), index=X_train.index)
        else:
            # 念のため DataFrame に変換（通常は来ない想定）
            X_train = pd.DataFrame(X_train)
            y_train = pd.Series(np.asarray(y_train), index=X_train.index)

        # 検証データも DataFrame を想定（experiment.py 側はそうなっている）
        if X_val is None or y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()
        else:
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)

        # ====== 2) Stage1（軽量化）：行・列サンプリング ======
        if self._stage1_mode:
            rs = self.seed

            # 行サンプリング
            X_train = X_train.sample(frac=self.stage1_row_frac, random_state=rs)
            y_train = y_train.loc[X_train.index]  # ← index で安全に揃える

            # 列サンプリング（検証側も同じ列に落とす）
            n_cols = X_train.shape[1]
            sel_cols = np.random.RandomState(rs).choice(
                X_train.columns,
                size=max(1, int(n_cols * self.stage1_col_frac)),
                replace=False,
            )
            X_train = X_train[sel_cols]
            X_val = X_val[sel_cols]

        # ====== 3) 学習・評価（固定閾値の F1） ======
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

        # 後片付け（メモリ回収）
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
        _model_config = self.model_config(
            trial,
            deepcopy(self.default_config),
            overrides=self.overrides,
            light=self._light_space,
        )
        if self.cv_n_splits >= 2:
            value = self.cross_validation(_model_config)
        else:
            value = self.one_shot(_model_config)
        trial.set_user_attr("f1", value)
        return value

    @staticmethod
    def _n_complete(study: optuna.Study) -> int:
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    @staticmethod
    def _build_storage(storage_path: Optional[str]):
        if storage_path is None:
            return None
        os.makedirs(storage_path, exist_ok=True)
        return optuna.storages.RDBStorage(url=f"sqlite:///{storage_path}/optuna.db")

    def _create_study(
        self, name: str, sampler: Optional[optuna.samplers.BaseSampler] = None
    ):
        storage_obj = self._build_storage(self.storage)
        return optuna.create_study(
            storage=storage_obj,
            study_name=name,
            direction="maximize",
            sampler=sampler
            or optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=self.n_startup_trials,
            ),
            load_if_exists=True,
        )

    def get_best_config(self) -> Dict[str, Any]:
        study = self._create_study(self.study_name)

        n_complete = self._n_complete(study)
        n_trials = max(0, int(self.n_trials) - n_complete)
        if n_trials > 0:
            logger.info(
                f"[Optuna] Running {n_trials} trial(s) (complete: {n_complete})"
            )
            study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)

        logger.info(f"[Optuna] Best F1={study.best_value:.6f}")
        logger.info(f"[Optuna] Best params={study.best_params}")
        update_model_config(self.default_config, study.best_params)
        return self.default_config


# =============================================================================
# 二段最適化クラス（oneshot → CV）
# =============================================================================


class OptimParamTwoStage(OptimParam):
    """
    Stage1: oneshot（軽い空間 + データサンプリング）
    Stage2: CV（top-K を元に縮小した空間で精査）
    """

    def __init__(
        self,
        *args,
        stage1_trials: int = 100,
        stage2_trials: int = 100,
        stage1_n_startup_trials: int = 20,
        stage2_n_startup_trials: int = 10,
        top_k: int = 15,  # Stage1 の上位K試行を参照
        shrink_factor: float = 0.5,  # 0<sf<=1 ほど狭く
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stage1_trials = int(stage1_trials)
        self.stage2_trials = int(stage2_trials)
        self.stage1_n_startup_trials = int(stage1_n_startup_trials)
        self.stage2_n_startup_trials = int(stage2_n_startup_trials)
        self.top_k = int(top_k)
        self.shrink_factor = float(shrink_factor)

    # ---- Stage 1: oneshot ----
    def _run_stage1(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=self.stage1_n_startup_trials
        )
        s1_name = f"{self.study_name}_stage1"
        s1 = self._create_study(s1_name, sampler=sampler)

        # oneshot & 軽量空間 & サンプリングを有効化
        prev_cv = self.cv_n_splits
        prev_overrides = self.overrides
        prev_light = self._light_space
        prev_stage1 = self._stage1_mode

        self.cv_n_splits = 0
        self.overrides = None
        self._light_space = True
        self._stage1_mode = True

        n_complete = self._n_complete(s1)
        n_trials = max(0, self.stage1_trials - n_complete)
        if n_trials > 0:
            logger.info(
                f"[TwoStage] Stage1 (oneshot) running {n_trials} trials (complete: {n_complete})."
            )
            s1.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)

        # restore
        self.cv_n_splits = prev_cv
        self.overrides = prev_overrides
        self._light_space = prev_light
        self._stage1_mode = prev_stage1

        return s1

    # ---- top-K から探索空間を縮小 ----
    @staticmethod
    def _guess_logscale(name: str) -> bool:
        return any(
            k in name
            for k in ["learning_rate", "reg_", "min_child_weight", "l2_leaf_reg"]
        )

    def _build_overrides_from_topk(
        self, study: optuna.Study
    ) -> Dict[str, Tuple[float, float, bool]]:
        trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trials.sort(
            key=lambda t: t.value if t.value is not None else -1.0, reverse=True
        )
        top = trials[: max(1, min(self.top_k, len(trials)))]

        dists = study.best_trial.distributions  # param -> distribution
        overrides: Dict[str, Tuple[float, float, bool]] = {}

        for pname, dist in dists.items():
            vals = [float(t.params[pname]) for t in top if pname in t.params]
            if not vals:
                continue
            vmin, vmax = min(vals), max(vals)
            if vmin == vmax:
                span = 1.0 if vmax == 0 else abs(vmax) * 0.1
                vmin, vmax = vmax - span, vmax + span

            # 元レンジ
            if hasattr(dist, "low") and hasattr(dist, "high"):
                low0, high0 = float(dist.low), float(dist.high)
            else:
                low0, high0 = vmin, vmax

            # 縮小（中心±幅×shrink）
            center = (vmin + vmax) / 2.0
            half_span = (vmax - vmin) / 2.0 * self.shrink_factor

            new_low = max(low0, center - half_span)
            new_high = min(high0, center + half_span)
            if new_low >= new_high:
                base_half = max(1e-8, (high0 - low0) * 0.05)
                new_low = max(low0, center - base_half)
                new_high = min(high0, center + base_half)

            log = getattr(dist, "log", False) or self._guess_logscale(pname)
            overrides[pname] = (new_low, new_high, log)

        return overrides

    # ---- Stage 2: CV ----
    def _run_stage2(
        self, overrides: Dict[str, Tuple[float, float, bool]]
    ) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=self.stage2_n_startup_trials
        )
        s2_name = f"{self.study_name}_stage2"
        s2 = self._create_study(s2_name, sampler=sampler)

        prev_cv = self.cv_n_splits
        prev_overrides = self.overrides
        prev_light = self._light_space

        self.cv_n_splits = max(5, prev_cv or 5)  # 少なくとも5-fold
        self.overrides = overrides
        self._light_space = False

        n_complete = self._n_complete(s2)
        n_trials = max(0, self.stage2_trials - n_complete)
        if n_trials > 0:
            logger.info(
                f"[TwoStage] Stage2 (CV={self.cv_n_splits}) running {n_trials} trials (complete: {n_complete})."
            )
            s2.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)

        # restore
        self.cv_n_splits = prev_cv
        self.overrides = prev_overrides
        self._light_space = prev_light

        return s2

    def get_best_config(self) -> Dict[str, Any]:
        # Stage1
        s1 = self._run_stage1()
        if len(s1.trials) == 0 or s1.best_trial is None:
            logger.warning(
                "[TwoStage] Stage1 produced no trials. Falling back to single-stage default search."
            )
            return super().get_best_config()

        # 上位から縮小空間
        overrides = self._build_overrides_from_topk(s1)
        logger.info(f"[TwoStage] Stage2 overrides (n={len(overrides)}): {overrides}")

        # Stage2
        s2 = self._run_stage2(overrides)

        # どちらが良いか
        if (
            len(s2.trials) > 0
            and s2.best_trial is not None
            and s2.best_value >= s1.best_value
        ):
            best = s2
            logger.info(f"[TwoStage] Best from Stage2 F1={best.best_value:.6f}")
        else:
            best = s1
            logger.info(f"[TwoStage] Best from Stage1 F1={best.best_value:.6f}")

        update_model_config(self.default_config, best.best_params)
        logger.info(f"[TwoStage] Best params={best.best_params}")
        return self.default_config
