from eda_tool.eda_core.data_quality import build_quality_report, check_duplicates
from eda_tool.eda_core.type_inference import profile_columns
import pandas as pd
from pathlib import Path


def test_duplicates_and_quality():
    fixture = Path(__file__).parent / 'fixtures' / 'sample.csv'
    df = pd.read_csv(fixture)
    dup = check_duplicates(df)
    assert dup['n_duplicate_rows'] == 0
    profiles = profile_columns(df)
    q = build_quality_report(df, profiles, 'target', 'classification')
    assert any(isinstance(f.column, str) for f in q)
