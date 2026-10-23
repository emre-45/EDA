import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr
from scipy.stats import contingency
from . import config
import pandas as pd


def _eta_squared_numeric_by_group(values: pd.Series, groups: pd.Series) -> float:
    df = pd.DataFrame({'v': values, 'g': groups}).dropna()
    if df.empty:
        return 0.0
    grand_mean = df['v'].mean()
    ss_total = ((df['v'] - grand_mean) ** 2).sum()
    ss_between = df.groupby('g').apply(lambda g: g.shape[0] * (g['v'].mean() - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total != 0 else 0.0


def compute_association(df: pd.DataFrame, feature_col: str, target_col: str,
                        feature_type: str, target_type: str, problem_type: str) -> dict:
    x = df[feature_col]
    y = df[target_col]
    # numeric-numeric
    if feature_type.startswith('numerical') and problem_type == 'regression':
        valid = x.dropna().index.intersection(y.dropna().index)
        if len(valid) < 2:
            return {'metric_name': 'pearson_r', 'metric_value': 0.0, 'figure': None}
        r, p = pearsonr(x.loc[valid], y.loc[valid])
        # If the numeric feature is discrete (low cardinality integers), render a violin per value
        try:
            nunique = int(x.loc[valid].nunique())
        except Exception:
            nunique = None
        is_integer_dtype = False
        try:
            is_integer_dtype = pd.api.types.is_integer_dtype(x.loc[valid])
        except Exception:
            # fallback: check values are integer-like
            try:
                vals = x.loc[valid].dropna().astype(float)
                is_integer_dtype = bool(np.allclose(vals, np.round(vals)))
            except Exception:
                is_integer_dtype = False

        if nunique is not None and nunique <= 20 and is_integer_dtype:
            # categorical-like numeric: violin per discrete value
            try:
                df_v = df.loc[valid].copy()
                # enforce numeric ordering of categories (avoid lexicographic ordering like '8' > '75')
                unique_vals = sorted(df_v[feature_col].dropna().unique(), key=lambda z: float(z))
                ordered_labels = [str(v) for v in unique_vals]
                df_v[feature_col] = df_v[feature_col].astype(str)
                fig = px.violin(df_v, x=feature_col, y=target_col, box=True, points='all', category_orders={feature_col: ordered_labels})
                fig.update_layout(margin=dict(l=40, r=20, t=20, b=40))
                return {'metric_name': 'pearson_r', 'metric_value': float(r), 'figure': fig}
            except Exception:
                pass

        # If the numeric feature has many unique values (continuous-like), bin into 5 equal-width bins
        try:
            if nunique is not None and nunique > 20:
                mn = float(np.nanmin(valid_x))
                mx = float(np.nanmax(valid_x))
                if mn == mx or np.isnan(mn) or np.isnan(mx):
                    # fallback to scatter if no range
                    pass
                else:
                    edges = np.linspace(mn, mx, num=6)
                    b = pd.cut(valid_x, bins=edges, include_lowest=True)
                    # keep ordered bin labels to avoid lexicographic ordering
                    labels = [str(c) for c in b.cat.categories]
                    df_bin = pd.DataFrame({feature_col: b.astype(str), target_col: valid_y})
                    fig = px.violin(df_bin, x=feature_col, y=target_col, box=True, points='all', category_orders={feature_col: labels})
                    fig.update_layout(margin=dict(l=40, r=20, t=20, b=40))
                    return {'metric_name': 'pearson_r', 'metric_value': float(r), 'figure': fig}
        except Exception:
            pass

        # Build figure manually: scatter points + single OLS line trace
        valid_x = x.loc[valid].astype(float)
        valid_y = y.loc[valid].astype(float)
        fig = go.Figure()
        # data points
        fig.add_trace(go.Scatter(x=valid_x, y=valid_y, mode='markers', marker=dict(size=10, opacity=0.95,
                                                                                  color='#636efa',
                                                                                  line=dict(width=0.6, color='black')), showlegend=False, name=''))
        # compute OLS line using numpy.polyfit
        try:
            coeffs = np.polyfit(valid_x, valid_y, deg=1)
            line_x = np.linspace(np.nanmin(valid_x), np.nanmax(valid_x), 200)
            line_y = coeffs[0] * line_x + coeffs[1]
            fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(width=2, color='rgba(200,30,30,0.9)'), showlegend=False, name='ols'))
        except Exception:
            # fallback: no OLS line
            pass
        fig.update_layout(margin=dict(l=40, r=20, t=20, b=40))
        return {'metric_name': 'pearson_r', 'metric_value': float(r), 'figure': fig}

    # numerical + classification -> eta-sq (numeric feature vs categorical target)
    if feature_type.startswith('numerical') and problem_type == 'classification':
        eta = _eta_squared_numeric_by_group(x, y)
        fig = px.box(df, x=y.name, y=feature_col)
        return {'metric_name': 'eta_squared', 'metric_value': float(eta), 'figure': fig}

    # categorical/boolean vs numerical (regression)
    if (feature_type.startswith('categorical') or feature_type == 'boolean') and problem_type == 'regression':
        eta = _eta_squared_numeric_by_group(y, x)
        fig = px.box(df, x=feature_col, y=target_col)
        return {'metric_name': 'eta_squared', 'metric_value': float(eta), 'figure': fig}

    # categorical/boolean vs categorical
    if (feature_type.startswith('categorical') or feature_type == 'boolean') and problem_type == 'classification':
        ct = pd.crosstab(df[feature_col], df[target_col])
        try:
            val = contingency.association(ct, method='cramer')
        except Exception:
            val = 0.0
        # normalized stacked bar
        prop = ct.div(ct.sum(axis=1), axis=0)
        # try numeric ordering of the category index when possible (avoid lexicographic ordering like '8' > '75')
        try:
            idx_num = pd.to_numeric(prop.index, errors='coerce')
            if idx_num.notna().sum() / max(1, len(idx_num)) >= 0.9:
                ordered_index = [x for _, x in sorted(zip(idx_num, prop.index))]
            else:
                ordered_index = list(prop.index.astype(str))
        except Exception:
            ordered_index = list(prop.index.astype(str))

        fig = go.Figure()
        for col in prop.columns:
            yvals = prop.reindex(ordered_index).loc[:, col].values
            fig.add_trace(go.Bar(name=str(col), x=ordered_index, y=yvals))
        fig.update_layout(barmode='stack')
        return {'metric_name': "cramers_v", 'metric_value': float(val), 'figure': fig}

    return {'metric_name': 'na', 'metric_value': 0.0, 'figure': None}


def rank_features_by_association(df: pd.DataFrame, profiles: list,
                                  target: str, problem_type: str) -> list[tuple[str, float]]:
    ranks = []
    target_profile = next((p for p in profiles if p.name == target), None)
    for p in profiles:
        if p.name == target or p.inferred_type in ('id_like', 'constant', 'free_text'):
            continue
        res = compute_association(df, p.name, target, p.inferred_type, target_profile.inferred_type if target_profile else '', problem_type)
        ranks.append((p.name, abs(float(res.get('metric_value', 0.0) or 0.0))))
    ranks.sort(key=lambda x: x[1], reverse=True)
    return ranks
