from dataclasses import dataclass
import pandas as pd
import numpy as np
from . import config
from typing import List


@dataclass
class QualityFlag:
    column: str
    severity: str
    message: str


def check_duplicates(df: pd.DataFrame) -> dict:
    n_dup = int(df.duplicated().sum())
    pct_dup = n_dup / df.shape[0] if df.shape[0]>0 else 0
    return {'n_duplicate_rows': n_dup, 'pct_duplicate': float(pct_dup)}


def detect_outliers_iqr(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {'n_outliers': 0, 'pct_outliers': 0.0, 'lower_bound': None, 'upper_bound': None}
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lb = q1 - config.OUTLIER_IQR_MULTIPLIER * iqr
    ub = q3 + config.OUTLIER_IQR_MULTIPLIER * iqr
    outliers = s[(s < lb) | (s > ub)]
    n_out = int(outliers.shape[0])
    pct = n_out / s.shape[0] if s.shape[0]>0 else 0
    return {'n_outliers': n_out, 'pct_outliers': float(pct), 'lower_bound': float(lb), 'upper_bound': float(ub)}


def check_class_imbalance(target_series: pd.Series, problem_type: str) -> dict | None:
    if problem_type != 'classification':
        return None
    s = target_series.dropna()
    props = (s.value_counts(normalize=True).to_dict())
    if len(props)==0:
        return {'class_proportions': {}, 'minority_class': None, 'minority_share': 0.0, 'is_imbalanced': False}
    minority_class = min(props, key=props.get)
    minority_share = props[minority_class]
    return {'class_proportions': props, 'minority_class': minority_class, 'minority_share': float(minority_share), 'is_imbalanced': minority_share < config.CLASS_IMBALANCE_THRESHOLD}


def build_quality_report(df: pd.DataFrame, profiles: List, target: str, problem_type: str) -> List[QualityFlag]:
    flags: List[QualityFlag] = []
    # duplicates
    dup = check_duplicates(df)
    if dup['n_duplicate_rows'] > 0:
        flags.append(QualityFlag(column='__dataset__', severity='warning', message=f"{dup['n_duplicate_rows']} duplicate rows"))

    for p in profiles:
        # critical
        if p.inferred_type == 'constant' or p.pct_missing > config.MISSING_CRITICAL_THRESHOLD:
            flags.append(QualityFlag(column=p.name, severity='critical', message='constant column' if p.inferred_type=='constant' else 'very high missingness'))
        # warning
        elif p.pct_missing > config.MISSING_WARNING_THRESHOLD:
            flags.append(QualityFlag(column=p.name, severity='warning', message='high missingness'))
        elif p.inferred_type == 'id_like':
            flags.append(QualityFlag(column=p.name, severity='warning', message='id-like column'))
        elif p.inferred_type == 'categorical_high':
            flags.append(QualityFlag(column=p.name, severity='warning', message='high cardinality categorical'))
        elif p.inferred_type == 'free_text':
            flags.append(QualityFlag(column=p.name, severity='info', message='free text column'))

        # numerical-specific checks
        if p.inferred_type.startswith('numerical'):
            s = df[p.name]
            try:
                import scipy.stats as ss
                sk = float(ss.skew(s.dropna())) if s.dropna().shape[0]>0 else 0.0
            except Exception:
                sk = 0.0
            if abs(sk) > config.SKEWNESS_FLAG_THRESHOLD:
                flags.append(QualityFlag(column=p.name, severity='warning', message='highly skewed'))
            out = detect_outliers_iqr(s)
            if out['pct_outliers'] > 0.05:
                flags.append(QualityFlag(column=p.name, severity='warning', message='many outliers'))

    # target checks
    if target in df.columns:
        if df[target].isna().all():
            flags.append(QualityFlag(column=target, severity='critical', message='target entirely missing'))
        # class imbalance
        ci = check_class_imbalance(df[target], problem_type)
        if ci and ci['is_imbalanced']:
            flags.append(QualityFlag(column=target, severity='warning', message='class imbalance'))

    # sort
    severity_order = {'critical': 0, 'warning': 1, 'info': 2}
    flags.sort(key=lambda x: (severity_order.get(x.severity, 3), x.column))
    return flags
