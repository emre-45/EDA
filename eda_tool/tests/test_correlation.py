from eda_tool.eda_core.correlation import flag_high_correlation_pairs, compute_vif
import pandas as pd
from pathlib import Path


def test_flag_high_corr():
    fixture = Path(__file__).parent / 'fixtures' / 'sample.csv'
    df = pd.read_csv(fixture)
    numeric = ['cont_num', 'disc_num']
    corr = df[numeric].corr()
    pairs = flag_high_correlation_pairs(corr)
    assert isinstance(pairs, list)
    # VIF likely None due to small rows
    v = compute_vif(df, numeric)
    assert v is None
