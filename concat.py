import pandas as pd
import numpy as np
import statistics as st
import re
from tabulate import tabulate

#
# train = pd.read_csv("datasets/train_fixed2.csv")
# test = pd.read_csv("datasets/test_fixed2.csv")

# train_concat = pd.read_csv("datasets/new_train_tfid_features.csv")
# test_concat = pd.read_csv("datasets/new_test_tfid_features.csv")
# train_concat = pd.read_csv("datasets/new_train_cntvec_features.csv")
# test_concat = pd.read_csv("datasets/new_test_cntvec_features.csv")

# train = pd.concat([train, train_concat], axis=1)
# test = pd.concat([test, test_concat], axis=1)

# train.to_csv("datasets/train_fixed3.csv", index=False)
# test.to_csv("datasets/test_fixed3.csv", index=False)

import os
import pandas as pd


def _read_feat(path: str) -> pd.DataFrame:
    """特徴量CSVを読み、id列名を小文字'id'に正規化して返す"""
    df = pd.read_csv(path)
    # id 列名の揺れを吸収
    id_cols = [c for c in df.columns if c.lower() == "id"]
    if len(id_cols) != 1:
        raise ValueError(
            f"[{path}] に id 列が見つからない/複数あります: {df.columns.tolist()}"
        )
    if id_cols[0] != "id":
        df = df.rename(columns={id_cols[0]: "id"})
    # 型をそろえる（'train_00001' などの文字列前提）
    df["id"] = df["id"].astype(str)
    # 重複チェック
    dup = df["id"].duplicated().sum()
    if dup:
        raise ValueError(f"[{path}] id 重複 {dup} 件あります")
    return df


def merge_features_on_id(
    base_train_csv: str,
    base_test_csv: str,
    feat_csvs_train: list[str],
    feat_csvs_test: list[str],
    out_train_csv: str,
    out_test_csv: str,
    fillna_value: float | int = 0.0,
    strict: bool = True,
):
    tr = pd.read_csv(base_train_csv)
    te = pd.read_csv(base_test_csv)

    # id を文字列 & 小文字名に（基データ側）
    id_cols_tr = [c for c in tr.columns if c.lower() == "id"]
    id_cols_te = [c for c in te.columns if c.lower() == "id"]
    assert (
        len(id_cols_tr) == 1 and len(id_cols_te) == 1
    ), "基データの id 列を確認してください"
    if id_cols_tr[0] != "id":
        tr = tr.rename(columns={id_cols_tr[0]: "id"})
    if id_cols_te[0] != "id":
        te = te.rename(columns={id_cols_te[0]: "id"})
    tr["id"] = tr["id"].astype(str)
    te["id"] = te["id"].astype(str)

    print(f"[base] train={tr.shape} test={te.shape}")

    # === train に train 特徴のみを結合 ===
    for p in feat_csvs_train:
        df = _read_feat(p)
        before_cols = set(tr.columns)
        m = tr.merge(df, on="id", how="left", indicator=True)
        miss = (m["_merge"] == "left_only").sum()
        if miss:
            print(f"  [WARN] {os.path.basename(p)}: train 側で未一致 {miss} 件")
        m = m.drop(columns=["_merge"])
        new_cols = [c for c in m.columns if c not in before_cols]
        # 追加列のみ欠損を埋める
        if new_cols:
            m[new_cols] = m[new_cols].fillna(fillna_value)
        tr = m
        print(
            f"  [train] +{os.path.basename(p)}  → {tr.shape}, added={len(new_cols)} cols"
        )

    # === test に test 特徴のみを結合 ===
    for p in feat_csvs_test:
        df = _read_feat(p)
        before_cols = set(te.columns)
        m = te.merge(df, on="id", how="left", indicator=True)
        miss = (m["_merge"] == "left_only").sum()
        if miss:
            print(f"  [WARN] {os.path.basename(p)}: test 側で未一致 {miss} 件")
        m = m.drop(columns=["_merge"])
        new_cols = [c for c in m.columns if c not in before_cols]
        if new_cols:
            m[new_cols] = m[new_cols].fillna(fillna_value)
        te = m
        print(
            f"  [test ] +{os.path.basename(p)}  → {te.shape}, added={len(new_cols)} cols"
        )

    # マージのループが終わった直後あたりに追記
    text_cols = ["name", "desc"]

    # 使わないならコメントアウト外して drop
    # tr = tr.drop(columns=[c for c in text_cols if c in tr.columns])
    # te = te.drop(columns=[c for c in text_cols if c in te.columns])

    # 残すなら空文字で埋める（←こちらが無難）
    for c in text_cols:
        if c in tr.columns:
            miss = tr[c].isna().sum()
            if miss:
                print(f"  [fix] train.{c} NaN → ''  ({miss} cells)")
            tr[c] = tr[c].fillna("")
        if c in te.columns:
            miss = te[c].isna().sum()
            if miss:
                print(f"  [fix] test.{c} NaN → ''  ({miss} cells)")
            te[c] = te[c].fillna("")

    # 最終NaNチェック（厳密）
    tr_nan = tr.isna().sum().sum()
    te_nan = te.isna().sum().sum()
    if strict and (tr_nan > 0 or te_nan > 0):
        bad_tr = tr.columns[tr.isna().any()].tolist()
        bad_te = te.columns[te.isna().any()].tolist()
        raise ValueError(
            f"NaN が残っています: train={tr_nan} cells ({bad_tr[:5]} ...), "
            f"test={te_nan} cells ({bad_te[:5]} ...)"
        )

    tr.to_csv(out_train_csv, index=False)
    te.to_csv(out_test_csv, index=False)
    print(
        f"[DONE] saved:\n  - {out_train_csv} ({tr.shape})\n  - {out_test_csv} ({te.shape})"
    )


feat_dir = "datasets"

# train用CSVだけ
feat_train = [
    f"{feat_dir}/train_name_tfidf_lsa_128d.csv",
    f"{feat_dir}/train_name_tfidf_nmf_64d.csv",
    f"{feat_dir}/train_desc_tfidf_lsa_128d.csv",
    f"{feat_dir}/train_desc_tfidf_nmf_64d.csv",
]

# test用CSVだけ（train_ を test_ に置換）
feat_test = [p.replace("train_", "test_") for p in feat_train]

merge_features_on_id(
    base_train_csv=f"{feat_dir}/train_fixed3.csv",
    base_test_csv=f"{feat_dir}/test_fixed3.csv",
    feat_csvs_train=feat_train,
    feat_csvs_test=feat_test,
    out_train_csv=f"{feat_dir}/train_fixed4.csv",
    out_test_csv=f"{feat_dir}/test_fixed4.csv",
    fillna_value=0.0,  # 追加した特徴列のみに適用
    strict=True,  # NaNが残っていれば例外を出す
)
