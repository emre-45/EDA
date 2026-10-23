from eda_tool.eda_core.type_inference import profile_columns
import pandas as pd
from pathlib import Path


def test_profiles_on_fixture():
    fixture = Path(__file__).parent / 'fixtures' / 'sample.csv'
    df = pd.read_csv(fixture)
    profiles = profile_columns(df)
    names = {p.name: p.inferred_type for p in profiles}
    assert names['const_col'] == 'constant'
    assert names['bool_col'] == 'boolean'
    assert names['dt_col'] == 'datetime'
    assert names['id_like_col'] == 'id_like'
    assert names['free_text'] in ('free_text', 'id_like', 'categorical_high', 'categorical_low')
