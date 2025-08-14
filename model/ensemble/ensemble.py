import numpy as np

from ..base_model import BaseRegressor, BaseClassifier
from ..gbm import (
    LightGBMRegressor,
    XGBoostRegressor,
    CatBoostRegressor,
    CatBoostClassifier,
    XGBoostClassifier,
    LightGBMClassifier,
)


class XGBLGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.xgb_model = XGBoostClassifier(
            input_dim,
            output_dim,
            model_config=self.model_config.xgboost,
            verbose=0,
            seed=seed,
        )
        self.lgbm_model = LightGBMClassifier(
            input_dim,
            output_dim,
            model_config=self.model_config.lightgbm,
            verbose=0,
            seed=seed,
        )

    def fit(self, X, y, eval_set):
        self.xgb_model.fit(X, y, eval_set)
        self.lgbm_model.fit(X, y, eval_set)

    def predict(self, X):
        # アンサンブルの割合
        n = 0.5
        return self.xgb_model.predict(X) * n + self.lgbm_model.predict(X) * (1 - n)

    def predict_proba(self, X):
        return (self.xgb_model.predict_proba(X) + self.lgbm_model.predict_proba(X)) / 2

    def feature_importance(self):
        return self.xgb_model.feature_importance(), self.lgbm_model.feature_importance()


# xgboost, lightgbm, catboostのアンサンブル
class XGBLGBMCATClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.xgb_model = XGBoostClassifier(
            input_dim,
            output_dim,
            model_config=self.model_config.xgboost,
            verbose=0,
            seed=seed,
        )
        self.lgbm_model = LightGBMClassifier(
            input_dim,
            output_dim,
            model_config=self.model_config.lightgbm,
            verbose=0,
            seed=seed,
        )
        self.cat_model = CatBoostClassifier(
            input_dim,
            output_dim,
            model_config=self.model_config.catboost,
            verbose=0,
            seed=seed,
        )

    def fit(self, X, y, eval_set):
        self.xgb_model.fit(X, y, eval_set)
        self.lgbm_model.fit(X, y, eval_set)
        self.cat_model.fit(X, y, eval_set)

    def predict(self, X):
        # アンサンブルの割合
        x = 0.333
        l = 0.333
        c = 1 - x - l
        return (
            self.xgb_model.predict(X) * x
            + self.lgbm_model.predict(X) * l
            + self.cat_model.predict(X) * c
        )

    def predict_proba(self, X):
        return (
            self.xgb_model.predict_proba(X)
            + self.lgbm_model.predict_proba(X)
            + self.cat_model.predict_proba(X)
        ) / 3

    def feature_importance(self):
        return (
            self.xgb_model.feature_importance(),
            self.lgbm_model.feature_importance(),
            self.cat_model.feature_importance(),
        )


# xgboost, lightgbmのアンサンブル
class XGBLGBMRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.xgb_model = XGBoostRegressor(
            input_dim,
            output_dim,
            model_config=self.model_config.xgboost,
            verbose=0,
            seed=seed,
        )
        self.lgbm_model = LightGBMRegressor(
            input_dim,
            output_dim,
            model_config=self.model_config.lightgbm,
            verbose=0,
            seed=seed,
        )

    def fit(self, X, y, eval_set):
        self.xgb_model.fit(X, y, eval_set)
        self.lgbm_model.fit(X, y, eval_set)

    def predict(self, X):
        # アンサンブルの割合
        n = 0.386
        return self.xgb_model.predict(X) * n + self.lgbm_model.predict(X) * (1 - n)

    def predict_proba(self, X):
        return (self.xgb_model.predict_proba(X) + self.lgbm_model.predict_proba(X)) / 2

    def feature_importance(self):
        return self.xgb_model.feature_importance(), self.lgbm_model.feature_importance()


# xgboost, lightgbm, catboostのアンサンブル
class XGBLGBMCATRegressor(BaseRegressor):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.xgb_model = XGBoostRegressor(
            input_dim,
            output_dim,
            model_config=self.model_config.xgboost,
            verbose=0,
            seed=seed,
        )
        self.lgbm_model = LightGBMRegressor(
            input_dim,
            output_dim,
            model_config=self.model_config.lightgbm,
            verbose=0,
            seed=seed,
        )
        self.cat_model = CatBoostRegressor(
            input_dim,
            output_dim,
            model_config=self.model_config.catboost,
            verbose=0,
            seed=seed,
        )

    def fit(self, X, y, eval_set):
        self.xgb_model.fit(X, y, eval_set)
        self.lgbm_model.fit(X, y, eval_set)
        self.cat_model.fit(X, y, eval_set)

    def predict(self, X):
        # アンサンブルの割合
        x = 0.317
        l = 0.395
        c = 1 - x - l
        return (
            self.xgb_model.predict(X) * x
            + self.lgbm_model.predict(X) * l
            + self.cat_model.predict(X) * c
        )

    def predict_proba(self, X):
        return (
            self.xgb_model.predict_proba(X)
            + self.lgbm_model.predict_proba(X)
            + self.cat_model.predict_proba(X)
        ) / 3

    def feature_importance(self):
        return (
            self.xgb_model.feature_importance(),
            self.lgbm_model.feature_importance(),
            self.cat_model.feature_importance(),
        )
