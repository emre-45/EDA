from eda_tool.eda_core.bivariate import compute_association
import pandas as pd
from pathlib import Path


def test_bivariate_metrics():
    fixture = Path(__file__).parent / 'fixtures' / 'sample.csv'
    df = pd.read_csv(fixture)
    # numeric-numeric
    r = compute_association(df, 'cont_num', 'disc_num', 'numerical_continuous', 'numerical_discrete', 'regression')
    assert r['metric_name'] == 'pearson_r'
    # categorical-categorical
    c = compute_association(df, 'cat_low', 'target', 'categorical_low', 'categorical_low', 'classification')
    assert c['metric_name'] in ('cramers_v', 'na')
