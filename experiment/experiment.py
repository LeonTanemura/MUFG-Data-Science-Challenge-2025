import logging
from time import time

import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

import dataset.dataset as dataset
from dataset import TabularDataFrame
from model import get_classifier, get_regressor

from .optuna import OptimParam
from .optuna_twostage import OptimParamTwoStage
from .utils import (
    cal_metrics_regression,
    load_json,
    set_seed,
    plot_confusion_matrix,
    concatenate_images,
)

from collections import Counter
import pickle

logger = logging.getLogger(__name__)


class ExpBase:
    def __init__(self, config):
        # seedの固定
        set_seed(config.seed)

        self.n_splits = config.n_splits
        self.model_name = config.model.name

        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        # train, testファイルの読み込み
        dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(
            seed=config.seed, **self.data_config
        )
        dfs = dataframe.processed_dataframes()
        self.categories_dict = dataframe.get_categories_dict()
        self.train, self.test = dfs["train"], dfs["test"]
        self.columns = dataframe.all_columns
        self.target_column = dataframe.target_column
        self.label_encoder = dataframe.label_encoder

        self.input_dim = len(self.columns)
        self.output_dim = len(self.label_encoder.classes_)

        self.id = dataframe.id

        self.seed = config.seed
        self.init_writer()

        self.save_model = config.save_model
        self.save_predict = config.save_predict
        self.existing_models = config.existing_models
        self.thresholds = config.thresholds

    # 評価指標
    def init_writer(self):
        metrics = [
            "fold",
            "ACC",
            "AUC",
            "Precision",
            "Recall",
            "F1",
        ]
        self.writer = {m: [] for m in metrics}

    # 予測評価値の追加
    def add_results(self, i_fold, scores: dict, time):
        self.writer["fold"].append(i_fold)
        for m in self.writer.keys():
            if m == "fold":
                continue
            self.writer[m].append(scores[m])

    # CV内で各モデルの作成
    def each_fold(self, i_fold, train_data, val_data):
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data)

        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            verbose=self.exp_config.verbose,
            seed=self.seed,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(
                val_data[self.columns],
                val_data[self.target_column].values.squeeze(),
            ),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end

    # 実行の基盤
    def run(self):
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        score_all = []
        # 作成済みモデルの使用（特殊なアンサンブルも実行）
        if self.existing_models:
            model_filename = "/home/leon/study/mydir/MUFG-Champion-Ship/outputs/single/main/V2/2024-08-31/15-52-41/lightgbm.pkl"
            with open(model_filename, "rb") as f:
                ex_models1 = pickle.load(f)
            model_filename = "/home/leon/study/mydir/MUFG-Champion-Ship/outputs/single/main/V2/2024-09-01/16-24-53/xgboost.pkl"
            with open(model_filename, "rb") as f:
                ex_models2 = pickle.load(f)
            ex_score_all = []
        models = []
        image_list = []
        # CVの実行
        for i_fold, (train_idx, val_idx) in enumerate(
            skf.split(self.train, self.train[self.target_column])
        ):
            if len(self.writer["fold"]) != 0 and self.writer["fold"][-1] >= i_fold:
                logger.info(f"Skip {i_fold + 1} fold. Already finished.")
                continue

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            # モデルの作成、追加
            model, time = self.each_fold(i_fold, train_data, val_data)
            models.append(model)

            # 各評価指標の値を算出
            score = cal_metrics_regression(
                model, val_data, self.columns, self.target_column
            )
            score.update(
                model.evaluate(
                    val_data[self.columns],
                    val_data[self.target_column].values.squeeze(),
                    self.thresholds[0],
                )
            )
            score_all.append(score)
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/ACC: {score['ACC']:.4f} | val/AUC: {score['AUC']:.4f} |"
                f" val/Precision: {score['Precision']:.4f} | val/Recall: {score['Recall']:.4f} | val/F1: {score['F1']:.4f} | Time: {time:.2f}s"
            )
            # confusion_matrixを計算してグラフを作成
            x_val, y_val = self.get_x_y(val_data)
            img = plot_confusion_matrix(model, x_val, y_val, i_fold)
            image_list.append(img)

            # 作成済みモデルの各評価指標の値を算出
            if self.existing_models:
                ex_score = cal_metrics_regression(
                    ex_models1[i_fold], val_data, self.columns, self.target_column
                )
                ex_score.update(
                    ex_models1[i_fold].evaluate(
                        val_data[self.columns],
                        val_data[self.target_column].values.squeeze(),
                    )
                )
                ex_score_all.append(ex_score)
                ex_score = cal_metrics_regression(
                    ex_models2[i_fold], val_data, self.columns, self.target_column
                )
                ex_score.update(
                    ex_models2[i_fold].evaluate(
                        val_data[self.columns],
                        val_data[self.target_column].values.squeeze(),
                    )
                )
                ex_score_all.append(ex_score)

        # 各foldのconfusion_matrixの合成
        concatenated_img = concatenate_images(image_list)
        if concatenated_img:
            concatenated_img.save("concatenated_confusion_matrices.png")

        # 各評価指標の平均値を算出
        final_score = Counter()
        for item in score_all:
            final_score.update(item)
        logger.info(
            f"[{self.model_name} results] ACC: {(final_score['ACC']/self.n_splits)} | AUC: {(final_score['AUC']/self.n_splits)} | "
            f"Precision: {(final_score['Precision']/self.n_splits)} | Recall: {(final_score['Recall']/self.n_splits)} | "
            f"F1: {(final_score['F1']/self.n_splits)}"
        )

        # 作成済みモデルを組み合わせた各評価指標の平均値を算出
        if self.existing_models:
            for item in ex_score_all:
                final_score.update(item)
            self.n_splits = self.n_splits * 3
            logger.info(
                f"[ensemble results] ACC: {(final_score['ACC']/self.n_splits)} | AUC: {(final_score['AUC']/self.n_splits)} | "
                f"Precision: {(final_score['Precision']/self.n_splits)} | Recall: {(final_score['Recall']/self.n_splits)} | "
                f"F1: {(final_score['F1']/self.n_splits)}"
            )
            for model in ex_models1:
                models.append(model)
            for model in ex_models2:
                models.append(model)

        # 各モデルの予測値を算出（self.thresholds を使用）
        # self.thresholds は config.thresholds から設定された閾値リスト
        global_thr = float(self.thresholds[0])
        logger.info(f"Using threshold for test prediction: {global_thr:.3f}")

        # モデルごとの陽性クラス確率を平均
        proba_list = []
        for model in models:
            proba = model.predict_proba(self.test[self.columns])[:, 1]
            proba_list.append(proba)
        mean_proba = np.mean(proba_list, axis=0)

        # 閾値で二値化
        predictions = (mean_proba >= global_thr).astype(int)
        logger.info(f"predictions: {predictions}")

        # 予測結果の保存
        if self.save_predict:
            pred = pd.DataFrame({"score": predictions})
            pred.to_csv(f"{self.model_name}_pred.csv", index=False)
        # モデルの保存
        if self.save_model:
            model_filename = f"{self.model_name}.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(models, f)

        # 提出ファイルの作成
        test_ids = (
            pd.DataFrame(self.id, columns=["id"])
            if isinstance(self.id, (list, pd.Series))
            else pd.DataFrame({"id": self.id})
        )
        submit_df = pd.concat([test_ids, pd.DataFrame({"score": predictions})], axis=1)
        submit_df.to_csv("submission.csv", index=False, header=False)

    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()

    def get_x_y(self, train_data):
        x, y = train_data[self.columns], train_data[self.target_column].values.squeeze()
        return x, y


# パラメータ探索をしない場合
class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)

    def get_model_config(self, *args, **kwargs):
        return self.model_config


# パラメータ探索をする場合
class ExpOptuna(ExpBase):
    def __init__(self, config):
        super().__init__(config)
        self.n_trials = config.exp.n_trials
        self.n_startup_trials = config.exp.n_startup_trials

        self.storage = config.exp.storage
        self.study_name = config.exp.study_name
        self.cv = config.exp.cv
        self.n_jobs = config.exp.n_jobs

    def run(self):
        if self.exp_config.delete_study:
            for i in range(self.n_splits):
                optuna.delete_study(
                    study_name=f"{self.exp_config.study_name}_{i}",
                    storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                )
                print(f"delete successful in {i}")
            return
        super().run()

    def get_model_config(self, i_fold, x, y, val_data):
        op = OptimParam(
            self.model_name,
            default_config=self.model_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            X=x,
            y=y,
            val_data=val_data,
            columns=self.columns,
            target_column=self.target_column,
            n_trials=self.n_trials,
            n_startup_trials=self.n_startup_trials,
            storage=self.storage,
            study_name=f"{self.study_name}_{i_fold}",
            cv=self.cv,
            n_jobs=self.n_jobs,
            seed=self.seed,
            threshold=float(self.thresholds[0]),
        )
        return op.get_best_config()


# パラメータ探索（二段・oneshot→CV）
class ExpOptunaTwoStage(ExpBase):
    def __init__(self, config):
        super().__init__(config)
        # 共通
        self.storage = self.exp_config.storage
        self.study_name = self.exp_config.study_name
        self.cv = self.exp_config.cv
        self.n_jobs = self.exp_config.n_jobs
        self.seed = self.seed  # 明示

        # 二段最適化専用パラメータ（未指定なら既定値で動作）
        self.stage1_trials = getattr(self.exp_config, "stage1_trials", 100)
        self.stage2_trials = getattr(self.exp_config, "stage2_trials", 100)
        self.stage1_n_startup_trials = getattr(
            self.exp_config, "stage1_n_startup_trials", 20
        )
        self.stage2_n_startup_trials = getattr(
            self.exp_config, "stage2_n_startup_trials", 10
        )
        self.top_k = getattr(self.exp_config, "top_k", 15)
        self.shrink_factor = getattr(self.exp_config, "shrink_factor", 0.5)

    def run(self):
        if self.exp_config.delete_study:
            # 二段は stage1/stage2 の2つの study を削除
            for i in range(self.n_splits):
                for suffix in ["_stage1", "_stage2"]:
                    try:
                        optuna.delete_study(
                            study_name=f"{self.exp_config.study_name}_{i}{suffix}",
                            storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                        )
                        print(f"delete successful: {i}{suffix}")
                    except Exception as e:
                        print(f"delete failed: {i}{suffix} -> {e}")
            return
        super().run()

    def get_model_config(self, i_fold, x, y, val_data):
        op = OptimParamTwoStage(
            self.model_name,
            default_config=self.model_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            X=x,
            y=y,
            val_data=val_data,
            columns=self.columns,
            target_column=self.target_column,
            n_trials=0,  # 二段クラスでは stage*_trials を使うため 0 でOK
            n_startup_trials=0,
            storage=self.storage,
            study_name=f"{self.study_name}_{i_fold}",
            cv=self.cv,  # Stage2で CV に昇格（内部で最低5-foldに調整）
            n_jobs=self.n_jobs,
            seed=self.seed,
            threshold=float(self.thresholds[0]),
            # 二段用
            stage1_trials=self.stage1_trials,
            stage2_trials=self.stage2_trials,
            stage1_n_startup_trials=self.stage1_n_startup_trials,
            stage2_n_startup_trials=self.stage2_n_startup_trials,
            top_k=self.top_k,
            shrink_factor=self.shrink_factor,
        )
        return op.get_best_config()
