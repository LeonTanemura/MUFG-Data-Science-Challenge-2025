import json
import operator as op
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
from omegaconf import OmegaConf


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data: Dict[str, Union[int, float, str]], save_dir: str = "./"):
    with open(os.path.join(save_dir, "results.json"), mode="wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path) -> Dict[str, Union[int, float, str]]:
    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_object(obj, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(input_path: str):
    with open(input_path, "rb") as f:
        return pickle.load(f)


# 各種スコアの計算（閾値指定可）
def cal_auc_score(model, data, feature_cols, label_col):
    """
    AUC スコアを計算。二値／多値どちらにも対応。
    """
    y_true = data[label_col].values
    pred_proba = model.predict_proba(data[feature_cols])
    if data[label_col].nunique() == 2:
        return roc_auc_score(y_true, pred_proba[:, 1])
    else:
        return roc_auc_score(y_true, pred_proba, multi_class="ovo")


def cal_acc_score(model, data, feature_cols, label_col, threshold=0.5):
    """
    Accuracy を計算。二値分類時は閾値で確率を二値化、多値分類時は predict をそのまま使用。
    """
    y_true = data[label_col].values
    if data[label_col].nunique() == 2:
        proba = model.predict_proba(data[feature_cols])[:, 1]
        y_pred = (proba >= threshold).astype(int)
    else:
        y_pred = model.predict(data[feature_cols])
    return accuracy_score(y_true, y_pred)


def cal_precision_score(model, data, feature_cols, label_col, threshold=0.5):
    """
    Precision を計算。二値分類時は threshold による二値化。
    multiclass では weighted average を使用。
    """
    y_true = data[label_col].values
    if data[label_col].nunique() == 2:
        proba = model.predict_proba(data[feature_cols])[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return precision_score(y_true, y_pred, average="binary", zero_division=0)
    else:
        y_pred = model.predict(data[feature_cols])
        return precision_score(y_true, y_pred, average="weighted")


def cal_recall_score(model, data, feature_cols, label_col, threshold=0.5):
    """
    Recall を計算。二値分類時は threshold による二値化。
    multiclass では weighted average を使用。
    """
    y_true = data[label_col].values
    if data[label_col].nunique() == 2:
        proba = model.predict_proba(data[feature_cols])[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return recall_score(y_true, y_pred, average="binary")
    else:
        y_pred = model.predict(data[feature_cols])
        return recall_score(y_true, y_pred, average="weighted")


def cal_f1_score(model, data, feature_cols, label_col, threshold=0.5):
    """
    F1 スコアを計算。二値分類時は threshold による二値化。
    multiclass では weighted average を使用。
    """
    y_true = data[label_col].values
    if data[label_col].nunique() == 2:
        proba = model.predict_proba(data[feature_cols])[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return f1_score(y_true, y_pred, average="binary", zero_division=0)
    else:
        y_pred = model.predict(data[feature_cols])
        return f1_score(y_true, y_pred, average="weighted")


def cal_metrics_regression(model, data, feature_cols, label_col):
    """
    各種評価指標をまとめて計算して dict で返却。
    threshold は二値分類時に適用。
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "conf", "main.yaml")
    config = OmegaConf.load(config_path)
    thresholds = list(config.thresholds)
    threshold = float(thresholds[0])
    return {
        "ACC": cal_acc_score(model, data, feature_cols, label_col, threshold),
        "AUC": cal_auc_score(model, data, feature_cols, label_col),
        "Precision": cal_precision_score(
            model, data, feature_cols, label_col, threshold
        ),
        "Recall": cal_recall_score(model, data, feature_cols, label_col, threshold),
        "F1": cal_f1_score(model, data, feature_cols, label_col, threshold),
    }


def set_categories_in_rule(ruleset, categories_dict):
    ruleset.set_categories(categories_dict)


def plot_confusion_matrix(model, x_val, y_val, i_fold):
    # 設定ファイルから閾値を取得（リストの最初の要素を使う）
    config_path = os.path.join(os.path.dirname(__file__), "..", "conf", "main.yaml")
    config = OmegaConf.load(config_path)
    thresholds = list(config.thresholds)
    threshold = float(thresholds[0])

    # クラス１の確率を取り出し、閾値で二値化
    proba = model.predict_proba(x_val)[:, 1]
    y_pred = (proba >= threshold).astype("int32")

    # 混同行列の計算
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    # グラフの作成
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"Confusion Matrix for fold {i_fold+1}")

    # 画像をバッファに保存して返却
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def concatenate_images(image_list):
    """
    画像リストを上下に連結した一枚の画像を返します。
    """
    if not image_list:
        return None

    widths, heights = zip(*(img.size for img in image_list))
    total_height = sum(heights)
    max_width = max(widths)

    concatenated_image = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in image_list:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    return concatenated_image
