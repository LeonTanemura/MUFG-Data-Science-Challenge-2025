import pandas as pd
import numpy as np
import statistics as st
import re
from tabulate import tabulate


# train2 = pd.read_csv("datasets/train_stacking_deberta_review_v2.csv")
# test2 = pd.read_csv("datasets/test_stacking_deberta_review_v2.csv")
# train3 = pd.read_csv("datasets/train_stacking_deberta_replyContent_v2.csv")
# test3 = pd.read_csv("datasets/test_stacking_deberta_replyContent_v2.csv")
# train4 = pd.read_csv("datasets/train_stacking_bert_review_v2.csv")
# test4 = pd.read_csv("datasets/test_stacking_bert_review_v2.csv")
# train5 = pd.read_csv("datasets/train_stacking_bert_replyContent_v2.csv")
# test5 = pd.read_csv("datasets/test_stacking_bert_replyContent_v2.csv")
# train6 = pd.read_csv("datasets/train_stacking_roberta_review_v2.csv")
# test6 = pd.read_csv("datasets/test_stacking_roberta_review_v2.csv")
# train7 = pd.read_csv("datasets/train_stacking_roberta_replyContent_v2.csv")
# test7 = pd.read_csv("datasets/test_stacking_roberta_replyContent_v2.csv")

# train['deberta_review_pred'] = train2['deberta_review_pred']
# test['deberta_review_pred'] = test2['deberta_review_pred']
# train['deberta_replyContent_pred'] = train3['deberta_replyContent_pred']
# test['deberta_replyContent_pred'] = test3['deberta_replyContent_pred']

# train['bert_review_pred'] = train4['bert_review_pred']
# test['bert_review_pred'] = test4['bert_review_pred']
# train['bert_replyContent_pred'] = train5['bert_replyContent_pred']
# test['bert_replyContent_pred'] = test5['bert_replyContent_pred']

# train['roberta_review_pred'] = train6['roberta_review_pred']
# test['roberta_review_pred'] = test6['roberta_review_pred']
# train['roberta_replyContent_pred'] = train7['roberta_replyContent_pred']
# test['roberta_replyContent_pred'] = test7['roberta_replyContent_pred']

# train8 = pd.read_csv("datasets/new_train_tfid_features.csv")
# test8 = pd.read_csv("datasets/new_test_tfid_features.csv")
# train9 = pd.read_csv("datasets/new_train_cntvec_features.csv")
# test9 = pd.read_csv("datasets/new_test_cntvec_features.csv")

# train = pd.concat([train, train8], axis=1)
# train = pd.concat([train, train9], axis=1)
# test = pd.concat([test, test8], axis=1)
# test = pd.concat([test, test9], axis=1)


# 欠損値の確認
def missing_value_checker(df, name):
    """データフレームの欠損値を確認して表示する"""
    chk_null = df.isnull().sum()
    chk_null_pct = chk_null / len(df)
    chk_null_tbl = pd.concat(
        [chk_null[chk_null > 0], chk_null_pct[chk_null > 0]], axis=1
    )
    chk_null_tbl = chk_null_tbl.rename(columns={0: "missing_count", 1: "missing_ratio"})

    print(f"--- {name} ---")
    if chk_null_tbl.empty:
        print("No missing values found.", end="\n\n")
    else:
        print(
            tabulate(chk_null_tbl, headers="keys", tablefmt="psql", floatfmt="f"),
            end="\n\n",
        )


def value_distribution_checker(df, max_unique=30):
    for col in df.columns:
        print(f"\n【特徴量名】{col}")
        unique_count = df[col].nunique(dropna=False)
        if unique_count <= max_unique:
            value_counts = df[col].value_counts(dropna=False)
            value_ratios = df[col].value_counts(normalize=True, dropna=False)
            summary = pd.concat([value_counts, value_ratios], axis=1).reset_index()
            summary.columns = [col, "sample size", "proportion"]
            print(
                tabulate(
                    summary,
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                    floatfmt="f",
                )
            )

        else:
            print("（数値データまたは一意の値が多いためスキップ）")


def check_dataset(train, test, train_test):
    missing_value_checker(train_test, "train_test")
    missing_value_checker(train, "train")
    missing_value_checker(test, "test")
    print("train_test dataset info:")
    print(train_test.info())
    print("\ntrain dataset info:")
    print(train.info())
    print("\ntest dataset info:")
    print(test.info())
    value_distribution_checker(train_test)


def simple_complementation(df, targets, method="mean"):
    for target in targets:
        if df[target].dtype == "object":
            # 文字列型の欠損値を最頻値で補完
            df[target] = df[target].fillna(df[target].mode()[0])
        else:
            # 数値型の欠損値を指定された方法で補完
            # mode(最頻値), mean(平均値), median(中央値)
            if method == "mean":
                df[target] = df[target].fillna(st.mean(df[target]))
            elif method == "median":
                df[target] = df[target].fillna(st.median(df[target]))
            elif method == "mode":
                df[target] = df[target].fillna(st.mode(df[target]))
            else:
                raise ValueError(
                    "Invalid method. Choose from 'mean', 'median', or 'mode'."
                )


# ===== 日時 → 数値特徴 =====
def add_time_features(df, use_state_changed=False):
    """
    created_at / launched_at / deadline（+任意で state_changed_at）を
    年・月・日・時・曜日・通し日・週・四半期・周期(sin/cos) などに分解。
    さらに 主要な差分: (launched - created), (deadline - launched) を追加。
    入力が 文字列 or UNIX秒 どちらでもOK。
    """
    df = df.copy()

    time_cols = ["created_at", "launched_at", "deadline"]
    if use_state_changed and "state_changed_at" in df.columns:
        time_cols += ["state_changed_at"]

    # 文字列/UNIX秒の両対応で datetime へ
    def _to_datetime(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_datetime(series, unit="s", errors="coerce")
        return pd.to_datetime(series, errors="coerce")

    for c in time_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")  # 型が何でも強制変換

    # 個別分解（NaT 安全）
    def decompose(prefix, s: pd.Series):
        # 1) 念のため毎回 datetime に統一（元が文字列でもOK）
        s = pd.to_datetime(s, errors="coerce")

        out = pd.DataFrame(index=s.index)

        # 2) UNIX秒の作り方を .view → .astype ベースに変更（NaTも安全に処理）
        ts_int = np.where(s.notna(), s.astype("int64") // 10**9, np.nan)
        out[f"{prefix}_ts"] = pd.Series(ts_int, index=s.index).astype("Int64")

        out[f"{prefix}_year"] = s.dt.year.astype("Int16")
        out[f"{prefix}_month"] = s.dt.month.astype("Int8")
        out[f"{prefix}_day"] = s.dt.day.astype("Int8")
        out[f"{prefix}_hour"] = s.dt.hour.astype("Int8")
        out[f"{prefix}_minute"] = s.dt.minute.astype("Int8")
        out[f"{prefix}_dow"] = s.dt.dayofweek.astype("Int8")
        out[f"{prefix}_doy"] = s.dt.dayofyear.astype("Int16")
        out[f"{prefix}_week"] = s.dt.isocalendar().week.astype("Int16")
        out[f"{prefix}_quarter"] = s.dt.quarter.astype("Int8")
        out[f"{prefix}_is_wkend"] = s.dt.dayofweek.isin([5, 6]).astype("Int8")
        out[f"{prefix}_is_month_start"] = s.dt.is_month_start.astype("Int8")
        out[f"{prefix}_is_month_end"] = s.dt.is_month_end.astype("Int8")
        out[f"{prefix}_is_qtr_end"] = s.dt.is_quarter_end.astype("Int8")
        out[f"{prefix}_is_year_end"] = s.dt.is_year_end.astype("Int8")

        out[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * out[f"{prefix}_hour"] / 24)
        out[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * out[f"{prefix}_hour"] / 24)
        out[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * out[f"{prefix}_dow"] / 7)
        out[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * out[f"{prefix}_dow"] / 7)
        out[f"{prefix}_mon_sin"] = np.sin(2 * np.pi * out[f"{prefix}_month"] / 12)
        out[f"{prefix}_mon_cos"] = np.cos(2 * np.pi * out[f"{prefix}_month"] / 12)
        return out

    for c in ["created_at", "launched_at", "deadline", "state_changed_at"]:
        if c in df.columns:
            df = pd.concat([df, decompose(c, df[c])], axis=1)

    # 主要な差分（秒→時間/日）
    if all(c in df.columns for c in ["launched_at", "created_at"]):
        diff = (df["launched_at"] - df["created_at"]).dt.total_seconds()
        df["prep_hours"] = (diff / 3600).astype("float32")
        df["prep_days"] = (diff / 86400).astype("float32")

    if all(c in df.columns for c in ["deadline", "launched_at"]):
        diff = (df["deadline"] - df["launched_at"]).dt.total_seconds()
        df["campaign_days"] = (diff / 86400).astype("float32")

    if use_state_changed and all(
        c in df.columns for c in ["state_changed_at", "launched_at"]
    ):
        diff = (df["state_changed_at"] - df["launched_at"]).dt.total_seconds()
        df["state_change_hours"] = (diff / 3600).astype("float32")

    # あり得ない負値は欠損へ
    for c in ["prep_hours", "prep_days", "campaign_days", "state_change_hours"]:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    return df


def add_prob_feature_by_id(
    df_all: pd.DataFrame,
    oof_csv: str,
    test_csv: str,
    id_col: str = "id",
    out_col: str = "prob_name",
    oof_col: str = "oof_prob_name",
    test_col: str = "prob_name",
) -> pd.DataFrame:
    """
    OOF用CSV（train側）と test予測CSV（test側）の確率を結合し、
    train+test を含む df_all に id で left join して out_col として追加する。
    """
    # 1) 読み込み（CSV想定。Excelなら read_excel に変更）
    oof = pd.read_csv(oof_csv, usecols=[id_col, oof_col])
    te = pd.read_csv(test_csv, usecols=[id_col, test_col])

    # 2) 列名を共通化
    oof = oof.rename(columns={oof_col: out_col})
    te = te.rename(columns={test_col: out_col})

    # 3) 結合用テーブル作成（id重複があれば最後を優先）
    mapper = pd.concat([oof, te], axis=0, ignore_index=True)
    dup_count = mapper[id_col].duplicated(keep=False).sum()
    if dup_count > 0:
        print(
            f"[WARN] {dup_count} rows have duplicated id in prob files. Keeping the last occurrence."
        )
        mapper = mapper.drop_duplicates(subset=[id_col], keep="last")

    # 4) 型整形
    mapper[out_col] = pd.to_numeric(mapper[out_col], errors="coerce").astype("float32")

    # 5) left join
    before_na = df_all[out_col].isna().sum() if out_col in df_all.columns else None
    df_all = df_all.merge(mapper, on=id_col, how="left")

    # 6) カバレッジ表示
    hit = df_all[out_col].notna().sum()
    miss = df_all[out_col].isna().sum()
    print(f"[{out_col}] merged: hit={hit:,}  miss={miss:,}  (total={len(df_all):,})")

    # 7) 既存列上書きのときの注意
    if before_na is not None:
        print(
            f"[{out_col}] existed before merge. NA before={before_na:,} / after={miss:,}"
        )

    return df_all


def preprocess(df):
    df["name"] = df["name"].fillna("").astype(str)
    df["desc"] = df["desc"].fillna("").astype(str)

    df = add_time_features(df, use_state_changed=False)

    return df


def create_csv(df):
    missing_value_checker(df, "train_test")
    train = df[df["id"].str.startswith("train_")]
    test = df[df["id"].str.startswith("test_")]
    test = test.drop("final_status", axis=1)
    train.to_csv("datasets/train_fixed.csv", index=False)
    test.to_csv("datasets/test_fixed.csv", index=False)


def main():
    OOF_NAME = "/home/leon/study/mydir/MUFG-Data-Science-Challenge-2025/datasets/oof_deberta_v3_large_name.csv"
    TEST_NAME = "/home/leon/study/mydir/MUFG-Data-Science-Challenge-2025/datasets/test_pred_deberta_v3_large_name.csv"
    OOF_DESC = "/home/leon/study/mydir/MUFG-Data-Science-Challenge-2025/datasets/oof_deberta_v3_base_desc.csv"
    TEST_DESC = "/home/leon/study/mydir/MUFG-Data-Science-Challenge-2025/datasets/test_pred_deberta_v3_base_desc.csv"
    train = pd.read_csv("datasets/new_train.csv")
    test = pd.read_csv("datasets/new_test.csv")
    train_test = pd.concat([train, test])
    check_dataset(train, test, train_test)
    train_test = preprocess(train_test)
    train_test = add_prob_feature_by_id(
        train_test,
        oof_csv=OOF_NAME,
        test_csv=TEST_NAME,
        id_col="id",
        out_col="prob_name",  # 追加したい列名
        oof_col="oof_prob_name",  # OOF 側の列名
        test_col="prob_name",  # test 側の列名
    )
    train_test = add_prob_feature_by_id(
        train_test,
        oof_csv=OOF_DESC,
        test_csv=TEST_DESC,
        id_col="id",
        out_col="prob_desc",  # 追加したい列名
        oof_col="oof_prob_desc",  # OOF 側の列名
        test_col="prob_desc",  # test 側の列名
    )
    df = train_test
    for i, c in enumerate(df.columns, 1):
        print(f"{i:02d}: {c:<30} {df[c].dtype}")

    create_csv(df)


if __name__ == "__main__":
    main()
