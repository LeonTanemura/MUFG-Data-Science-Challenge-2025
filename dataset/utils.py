import numpy as np
import pandas as pd


def feature_name_combiner(col, value) -> str:
    def replace(s):
        return s.replace("<", "lt_").replace(">", "gt_").replace("=", "eq_").replace("[", "lb_").replace("]", "ub_")

    col = replace(str(col))
    value = replace(str(value))
    return f'{col}="{value}"'


def feature_name_restorer(feature_name) -> str:
    return (
        feature_name.replace("lt_", "<").replace("gt_", ">").replace("eq_", "=").replace("lb_", "[").replace("ub_", "]")
    )


def label_encode(y: pd.Series):
    value_counts = y.value_counts(normalize=True)
    label_mapping = {value: index for index, (value, _) in enumerate(value_counts.items())}
    y_labels = y.map(label_mapping).astype(np.int32)
    return y_labels