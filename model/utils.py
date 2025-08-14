import numpy as np
from sklearn.metrics import f1_score, log_loss, cohen_kappa_score
import lightgbm as lgb
import xgboost as xgb


def feval_f1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    XGBoost の train()/fit() に渡すカスタム評価関数。
    preds: モデルから出力される連続的な予測値 (probabilities)
    dtrain: xgb.DMatrix オブジェクト (ラベルは dtrain.get_label() で取得)
    """
    y_true = y_true.astype(int)
    y_pred_label = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred_label, average="micro", zero_division=0)


def f1_micro(preds, dtrain):
    # XGBoost の feval 用シグネチャ：まず preds（連続値）、次に DMatrix
    y_true = dtrain.get_label().astype(int)
    # 0.5 を閾値にしてバイナリ予測に変換
    y_pred = (preds > 0.5).astype(int)
    score = f1_score(y_true, y_pred, average="micro", zero_division=0)
    # XGBoost には (名前, 値) のタプルで返す
    return "f1_micro", score


def f1_micro_lgb(y_true, y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return "f1_micro", f1_score(y_true, y_pred, average="micro", zero_division=0), True


def binary_logloss(y_true, y_pred):
    return "binary_logloss", log_loss(y_true, y_pred), False


# 今回 a は未使用
def quadratic_weighted_kappa(y_true, y_pred):
    a = 0
    # y_pred が XGB の QuantileDMatrix インスタンスかどうかを確認
    if isinstance(y_pred, xgb.QuantileDMatrix):
        # XGB の場合
        # y_true と y_pred を入れ替える
        y_true, y_pred = y_pred, y_true
        # ラベルデータを取得し、四捨五入
        y_true = (y_true.get_label() + a).round()
        # 予測値をクリップして四捨五入
        y_pred = (y_pred + a).clip(0, 4).round()
        # QWKスコアを計算
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return "QWK", qwk

    else:
        # LightGBM 用の処理
        # ラベルに a を加算
        y_true = y_true + a
        # 予測値をクリップして四捨五入
        y_pred = (y_pred + a).clip(0, 4).round()
        # QWKスコアを計算
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    a = 0
    b = 1
    # ラベルと予測値に a を加算
    labels = y_true + a
    preds = y_pred + a
    # 予測値をクリップ
    preds = preds.clip(0, 4)
    # 損失関数 f と g を計算
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - a) ** 2 + b)
    # 勾配とヘッセを計算
    df = preds - labels
    dg = preds - a
    grad = (df / g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))
    return grad, hess


# a = 2.948
# b = 1.092
