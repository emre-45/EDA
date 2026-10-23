import pandas as pd
from .data_quality import QualityFlag
from . import config


def recommend_preprocessing(profile, is_flagged_skewed: bool, is_flagged_high_corr: bool) -> list[str]:
    recs = []
    it = profile.inferred_type
    if it == 'categorical_low':
        recs.append('One-hot encode')
    if it == 'categorical_high':
        recs.append('High cardinality: use target or frequency encoding, not one-hot')
    if it in ('numerical_continuous', 'numerical_discrete') and is_flagged_skewed:
        recs.append('Highly skewed: consider log or Box-Cox transform')
    if it.startswith('numerical'):
        recs.append('Scale (standardize or normalize) before distance-based models')
    if profile.pct_missing > 0 and it.startswith('numerical'):
        recs.append('Impute missing values (median recommended over mean given skew check above)')
    if profile.pct_missing > 0 and it.startswith('categorical'):
        recs.append("Impute missing values (mode, or add an explicit 'Missing' category)")
    if it == 'id_like':
        recs.append('Drop before modeling (identifier column, no predictive signal)')
    if it == 'constant':
        recs.append('Drop before modeling (zero variance)')
    if it == 'free_text':
        recs.append('Not modeling-ready as-is: requires NLP feature extraction (TF-IDF, embeddings) or drop')
    if is_flagged_high_corr:
        recs.append('Involved in a high-correlation pair — consider dropping one of the two before modeling')
    return recs


def build_ml_readiness_table(profiles, quality_flags, high_corr_pairs):
    # determine which columns are flagged as skewed or high_corr
    high_corr_cols = set([a for a, b, _ in high_corr_pairs] + [b for a, b, _ in high_corr_pairs]) if high_corr_pairs else set()
    skewed = {f.column for f in quality_flags if 'skew' in f.message}
    rows = []
    for p in profiles:
        if p.name == '':
            continue
        recs = recommend_preprocessing(p, p.name in skewed, p.name in high_corr_cols)
        rows.append({'column': p.name, 'inferred_type': p.inferred_type, 'recommendations': '; '.join(recs)})
    return pd.DataFrame(rows)
