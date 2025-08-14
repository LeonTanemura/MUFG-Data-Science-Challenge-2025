from experiment.utils import set_seed

from .gbm import (
    CatBoostClassifier,
    LightGBMClassifier,
    XGBoostClassifier,
    LightGBMRegressor,
    XGBoostRegressor,
    CatBoostRegressor,
)
from .ensemble import (
    XGBLGBMClassifier,
    XGBLGBMCATClassifier,
    XGBLGBMRegressor,
    XGBLGBMCATRegressor,
)


def get_classifier(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "lightgbm":
        return LightGBMClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "catboost":
        return CatBoostClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "xgblgbm":
        return XGBLGBMClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "xgblgbmcat":
        return XGBLGBMCATClassifier(input_dim, output_dim, model_config, verbose, seed)
    else:
        raise KeyError(f"{name} is not defined.")


# yamlファイルで指定された名前から使用するモデルの呼び出し
def get_regressor(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "lightgbm":
        return LightGBMRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "catboost":
        return CatBoostRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "xgblgbm":
        return XGBLGBMRegressor(input_dim, output_dim, model_config, verbose, seed)
    elif name == "xgblgbmcat":
        return XGBLGBMCATRegressor(input_dim, output_dim, model_config, verbose, seed)
    else:
        raise KeyError(f"{name} is not defined.")
