import pandas as pd
from scipy.stats import shapiro, chi2_contingency, f_oneway
from . import config
from typing import Optional


def test_normality(series: pd.Series) -> dict | None:
    s = series.dropna()
    if s.shape[0] < 3:
        return None
    sample = s
    if s.shape[0] > config.SHAPIRO_MAX_N:
        sample = s.sample(config.SHAPIRO_MAX_N, random_state=config.RANDOM_SEED)
    stat, p = shapiro(sample)
    return {'statistic': float(stat), 'p_value': float(p), 'is_normal': p > config.ALPHA}


def test_chi_square(cat1: pd.Series, cat2: pd.Series) -> dict | None:
    ct = pd.crosstab(cat1, cat2)
    if ct.size == 0:
        return None
    chi2, p, dof, expected = chi2_contingency(ct)
    if (expected < 5).any():
        return {'reason': 'low_expected_counts'}
    return {'statistic': float(chi2), 'p_value': float(p), 'dof': int(dof), 'significant': p < config.ALPHA}


def test_anova(numeric: pd.Series, group: pd.Series) -> dict | None:
    groups = [numeric[group == g].dropna() for g in group.dropna().unique()]
    groups = [g for g in groups if g.shape[0] >= 2]
    if len(groups) < 2:
        return None
    f_stat, p = f_oneway(*groups)
    return {'statistic': float(f_stat), 'p_value': float(p), 'significant': p < config.ALPHA}


def run_all_applicable_tests(df: pd.DataFrame, profiles: list, target: str, problem_type: str) -> pd.DataFrame:
    # Statistical tests disabled globally per user request.
    # Return empty DataFrame so report omits Statistical Tests section.
    return pd.DataFrame([])
