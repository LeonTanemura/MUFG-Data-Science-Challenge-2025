# MUFG Data Science Challenge 2025

## 概要
MUFGデータサイエンスチャレンジ2025のソリューションコード

## 環境設定
### 必要条件
- Python 3.10.12
- 必要なライブラリ

```bash
pip install -r requirements.txt
```

## プロジェクト構造
```
├── conf/           # 設定ファイル
├── dataset/        # データセット関連のコード
├── datasets/       # 生データと処理済みデータ
├── experiment/     # 実験用コード
├── model/         # モデル定義
└── outputs/       # 学習結果の出力
```

## データの準備
1. `datasets`ディレクトリを作成
2. 訓練データ(`train.csv`)とテストデータ(`test.csv`)を配置
3. 前処理の実行:
```bash
python preprocess.py
```

## 学習の実行
```bash
python main.py
```

## 主な機能
- カテゴリカル変数のエンコーディング
- テキストデータの特徴量エンジニアリング
- 複数のモデル（LightGBM, XGBoost, CatBoost）による学習
- Optunaによるハイパーパラメータ最適化