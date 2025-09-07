import lightgbm as lgb
import xgboost as xgb
from lightgbm.callback import log_evaluation, early_stopping
import catboost
from sklearn.utils.validation import check_X_y
import numpy as np
import pandas as pd

from .base_model import BaseClassifier, BaseRegressor
from .utils import (
    feval_f1,
    f1_micro,
    f1_micro_lgb,
    binary_logloss,
    qwk_obj,
    quadratic_weighted_kappa,
)


class XGBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",  # 2値分類用のobjectiveに変更
            # num_class=self.output_dim,
            eval_metric=["logloss", "auc"],
            early_stopping_rounds=50,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = check_X_y(*eval_set)

        self.model.fit(X, y, eval_set=[eval_set], verbose=self.verbose > 0)

    def feature_importance(self):
        return self.model.feature_importances_


class LightGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMClassifier(
            objective="binary",  # 2値分類用のobjectiveに変更
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = check_X_y(*eval_set)

        self.model.fit(
            X,
            y,
            eval_set=[eval_set],
            eval_metric=["binary_logloss", "auc"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=self.verbose > 0),
                lgb.log_evaluation(period=100),
            ],
        )
        # print("LightGBMモデルの使用されたパラメータ:", self.model.get_params())

    def feature_importance(self):
        return self.model.feature_importances_


class CatBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None):
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = catboost.CatBoostClassifier(
            loss_function="Logloss",  # 目的関数
            eval_metric="Logloss",  # モデル選択の主評価指標
            custom_metric=["AUC"],  # 追加で出力したい指標
            random_seed=seed,
            verbose=self.verbose > 0,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        X_val, y_val = eval_set
        X_val, y_val = check_X_y(X_val, y_val)

        self.model.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=self.verbose > 0,
        )

    def feature_importance(self):
        return self.model.get_feature_importance()


# 未使用
class XGBoostRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",  # 目的関数を回帰用に変更
            **self.model_config,
            random_state=seed,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        print(self._column_names)
        print(X.head())
        X, y = check_X_y(X, y)
        eval_set = [
            (X_val.values if isinstance(X_val, pd.DataFrame) else X_val, y_val)
            for X_val, y_val in eval_set
        ]

        xgb_callbacks = [
            xgb.callback.EvaluationMonitor(period=10000),
            xgb.callback.EarlyStopping(
                100, metric_name="QWK", maximize=True, save_best=True
            ),
        ]

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=f1_micro,  # 評価指標を指定
            callbacks=xgb_callbacks,
            verbose=0,
        )

    def feature_importance(self):
        return self.model.feature_importances_


# 未使用
class LightGBMRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMRegressor(
            objective=qwk_obj,  # 目的関数のカスタム
            # objective='regression',
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)

        eval_set = [
            (X_val.values if isinstance(X_val, pd.DataFrame) else X_val, y_val)
            for X_val, y_val in eval_set
        ]

        self.model.fit(
            X,
            y,
            eval_names=["train", "valid"],
            eval_set=eval_set,
            eval_metric=quadratic_weighted_kappa,  # 評価指標を指定
            callbacks=[early_stopping(stopping_rounds=100)],
            # callbacks=[log_evaluation(period=100), early_stopping(stopping_rounds=75)],
        )

    def feature_importance(self):
        return self.model.feature_importances_


# 未使用
class CatBoostRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = catboost.CatBoostRegressor(
            loss_function=qwk_obj,  # QWKを目的関数として設定
            random_seed=seed,
            objective="RMSE",
            eval_metric="RMSE",
            **self.model_config,
            verbose=0,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)

        # `Pool`オブジェクトにデータを分割して渡す
        train_pool = catboost.Pool(X, y)
        eval_pools = [
            catboost.Pool(
                X_val.values if isinstance(X_val, pd.DataFrame) else X_val, y_val
            )
            for X_val, y_val in eval_set
        ]

        self.model.fit(
            train_pool,
            eval_set=eval_pools,
            use_best_model=True,
            early_stopping_rounds=75,
        )

        # 結果を表示
        best_iteration = self.model.get_best_iteration()
        best_score = self.model.get_best_score()
        print(f"Best iteration: {best_iteration}")
        print(f"Best score: {best_score}")

    def feature_importance(self):
        return self.model.get_feature_importance()
