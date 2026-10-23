import pandas as pd
from . import config


def compute_correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    return df[numeric_cols].corr(method='pearson')


def flag_high_correlation_pairs(corr_matrix: pd.DataFrame) -> list[tuple[str, str, float]]:
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a = cols[i]; b = cols[j]
            val = corr_matrix.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(val) > config.CORRELATION_FLAG_THRESHOLD:
                pairs.append((a, b, float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def compute_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame | None:
    # VIF computation disabled globally per user request.
    # Returning None so report generation omits VIF output.
    return None
