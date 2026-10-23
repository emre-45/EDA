from dataclasses import dataclass
import pandas as pd
import numpy as np
from . import config


@dataclass
class ColumnProfile:
    name: str
    original_dtype: str
    inferred_type: str
    n_missing: int
    pct_missing: float
    n_unique: int
    unique_ratio: float
    sample_values: list


def _is_bool_like(series: pd.Series) -> bool:
    if series.dtype == bool:
        return True
    if pd.api.types.is_numeric_dtype(series):
        uniques = series.dropna().unique()
        if len(uniques) == 2 and set(uniques).issubset({0, 1}):
            return True
    return False


def infer_column_type(series: pd.Series, n_rows: int) -> str:
    non_null = series.dropna()
    nunique = non_null.nunique(dropna=True)
    count_non_null = non_null.shape[0]

    # 1 Constant
    if nunique <= 1:
        return 'constant'

    # 2 Boolean
    if _is_bool_like(series):
        return 'boolean'

    # 3 Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = non_null.sample(min(len(non_null), config.DATETIME_PARSE_SAMPLE_SIZE), random_state=config.RANDOM_SEED) if len(non_null)>0 else non_null
        parsed = pd.to_datetime(sample, errors='coerce')
        success = parsed.notna().sum()
        if len(sample) > 0 and (success / len(sample)) >= config.DATETIME_PARSE_SUCCESS_THRESHOLD:
            return 'datetime'

    # 4 ID-like
    if n_rows >= config.ID_LIKE_MIN_ROWS:
        unique_ratio = nunique / count_non_null if count_non_null > 0 else 0
        # For non-numeric series, a high unique ratio indicates an ID-like column.
        if not pd.api.types.is_numeric_dtype(series):
            if unique_ratio > config.ID_LIKE_UNIQUE_RATIO:
                return 'id_like'

        # For numeric series, be conservative: only mark as id_like when
        # values are integer-like and appear to be a contiguous integer
        # sequence (e.g., an auto-incrementing index). This avoids treating
        # per-sample numeric metrics (floats or non-sequential ints) as IDs.
        if pd.api.types.is_numeric_dtype(series) and unique_ratio > config.ID_LIKE_UNIQUE_RATIO:
            dropped = non_null.dropna()
            # detect integer-like (either integer dtype or all values have zero fractional part)
            integer_like = False
            if pd.api.types.is_integer_dtype(dropped):
                integer_like = True
            else:
                try:
                    integer_like = (dropped.astype(float) % 1 == 0).all()
                except Exception:
                    integer_like = False

            if integer_like and dropped.shape[0] > 0:
                try:
                    vals = dropped.astype(int)
                    mn = int(vals.min())
                    mx = int(vals.max())
                    # contiguous range check
                    if (mx - mn + 1) == nunique:
                        return 'id_like'
                except Exception:
                    pass

    # 5 Numerical
    if pd.api.types.is_numeric_dtype(series) and nunique > config.NUMERIC_CATEGORICAL_MAX_UNIQUE:
        if pd.api.types.is_integer_dtype(series):
            return 'numerical_discrete'
        return 'numerical_continuous'

    # 6 Numeric categorical
    if pd.api.types.is_numeric_dtype(series) and nunique <= config.NUMERIC_CATEGORICAL_MAX_UNIQUE:
        return 'categorical_low'

    # Object/category types
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        mean_len = non_null.astype(str).map(len).mean() if count_non_null>0 else 0
        if nunique <= config.HIGH_CARDINALITY_THRESHOLD:
            return 'categorical_low'
        if nunique > config.HIGH_CARDINALITY_THRESHOLD and mean_len <= config.TEXT_MIN_AVG_LENGTH:
            return 'categorical_high'
        if nunique > config.HIGH_CARDINALITY_THRESHOLD and mean_len > config.TEXT_MIN_AVG_LENGTH:
            return 'free_text'

    # Fallback
    return 'categorical_low'


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    profiles = []
    n_rows = df.shape[0]
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        n_missing = s.isna().sum()
        pct_missing = n_missing / n_rows if n_rows>0 else 0
        n_unique = non_null.nunique(dropna=True)
        unique_ratio = n_unique / non_null.shape[0] if non_null.shape[0]>0 else 0
        inferred = infer_column_type(s, n_rows)
        sample_values = list(non_null.head(5).astype(str).tolist())
        profiles.append(ColumnProfile(name=col,
                                      original_dtype=str(s.dtype),
                                      inferred_type=inferred,
                                      n_missing=int(n_missing),
                                      pct_missing=float(pct_missing),
                                      n_unique=int(n_unique),
                                      unique_ratio=float(unique_ratio),
                                      sample_values=sample_values))
    return profiles


def infer_problem_type(target_profile: ColumnProfile) -> str:
    if target_profile.inferred_type.startswith('numerical'):
        return 'regression'
    return 'classification'
