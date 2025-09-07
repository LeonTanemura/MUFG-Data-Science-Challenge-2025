from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    cohen_kappa_score,
)
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import os


class BaseClassifier:
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        self.verbose = verbose

    def fit(self, X, y, eval_set):
        raise NotImplementedError()

    def predict_proba(self, X):
        return self.model.predict_proba(X.values)

    def predict(self, X):
        return self.model.predict(X.values)

    def evaluate(self, X, y, threshold=0.5):
        y_score = self.predict_proba(X)[:, 1]
        y_pred = (y_score >= threshold).astype(int)

        # 2. 各種評価指標の計算
        results = {
            "ACC": accuracy_score(y, y_pred),
            "AUC": roc_auc_score(y, y_score),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "Specificity": recall_score(1 - y, 1 - y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0),
        }
        return results

    def feature_importance(self):
        raise NotImplementedError()


# 未使用
class BaseRegressor:
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        self.verbose = verbose

    def fit(self, X, y, eval_set):
        raise NotImplementedError()

    def predict(self, X):
        return self.model.predict(X.values)

    def evaluate(self, X, y):
        # thresholds を取得
        config_path = os.path.join(os.path.dirname(__file__), "..", "conf", "main.yaml")
        config = OmegaConf.load(config_path)
        thresholds = config.thresholds

        y_pred = self.predict(X)
        results = {}
        # 各評価指標の値を算出
        mse = mean_squared_error(y, y_pred)
        results["MSE"] = mse
        results["MAE"] = mean_absolute_error(y, y_pred)
        results["RMSE"] = np.sqrt(mse)
        y_pred = pd.cut(
            y_pred, bins=[-np.inf] + thresholds + [np.inf], labels=[0, 1, 2, 3, 4]
        ).astype(int)
        results["QWK"] = cohen_kappa_score(y, y_pred, weights="quadratic")
        return results

    def feature_importance(self):
        raise NotImplementedError()
