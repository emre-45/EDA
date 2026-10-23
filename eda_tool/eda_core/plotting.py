import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from . import config
import base64
from typing import Any


def plot_histogram_box(series: pd.Series, column_name: str) -> go.Figure:
    s = series.dropna()
    # attempt to coerce to numeric for consistent behavior
    num = pd.to_numeric(s, errors='coerce')
    num_frac = num.notna().sum() / max(1, s.shape[0])
    if num_frac >= 0.9:
        # numeric-like: use histogram
        fig = px.histogram(num, x=series.name, marginal='box')
    else:
        # non-numeric: fallback to value counts bar
        vc = s.astype(str).value_counts()
        fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x': column_name, 'y': 'count'})
    fig.update_layout(title=column_name)
    return fig


def plot_bar_categorical(series: pd.Series, column_name: str, top_n: int = config.TOP_N_CATEGORIES) -> go.Figure:
    s = series.dropna()
    # try coercing to numeric to decide ordering/binning
    num = pd.to_numeric(s, errors='coerce')
    num_frac = num.notna().sum() / max(1, s.shape[0])

    # If mostly numeric, treat specially
    if num_frac >= 0.9:
        nunique = int(num.nunique())
        mn, mx = num.min(), num.max()
        # If many unique values, bin them (with age-aware bins when plausible)
        if nunique > top_n:
            # create 5 equal-width bins between min and max
            try:
                import numpy as _np
                # if mn == mx, fallback to single bin
                if pd.isna(mn) or pd.isna(mx) or mn == mx:
                    b = pd.cut(num, bins=1)
                else:
                    edges = _np.linspace(mn, mx, num=6)
                    b = pd.cut(num, bins=edges, include_lowest=True)
                vc = b.value_counts().sort_index()
                labels = [str(i) for i in vc.index]
                fig = px.bar(x=labels, y=vc.values, labels={'x': column_name, 'y': 'count'})
                fig.update_layout(title=column_name)
                return fig
            except Exception:
                pass
        # otherwise small set of numeric categories: show sorted numeric bars
        vc = num.value_counts().sort_index()
        fig = px.bar(x=[str(x) for x in vc.index], y=vc.values, labels={'x': column_name, 'y': 'count'})
        fig.update_layout(title=column_name)
        return fig

    # non-numeric: default categorical handling with top N
    vc = s.astype(str).value_counts()
    if vc.shape[0] > top_n:
        top = vc.iloc[:top_n]
        other = vc.iloc[top_n:].sum()
        top['Other'] = other
        vc = top
    fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={'x': column_name, 'y': 'count'})
    fig.update_layout(title=column_name)
    return fig


def plot_bar_boolean(series: pd.Series, column_name: str) -> go.Figure:
    vc = series.dropna().astype(str).value_counts()
    fig = px.bar(x=vc.index, y=vc.values, labels={'x': column_name, 'y': 'count'})
    fig.update_layout(title=column_name)
    return fig


def plot_datetime_trend(series: pd.Series, column_name: str) -> go.Figure:
    s = pd.to_datetime(series.dropna())
    if s.empty:
        return go.Figure()
    span = (s.max() - s.min()).days
    if span <= 90:
        freq = 'D'
    elif span <= 730:
        freq = 'W'
    else:
        freq = 'M'
    counts = s.dt.to_period(freq).value_counts().sort_index()
    fig = px.line(x=[str(x) for x in counts.index], y=counts.values)
    fig.update_layout(title=column_name)
    return fig


def plot_missing_matrix(df: pd.DataFrame) -> go.Figure:
    m = df.isna().astype(int)
    fig = px.imshow(m.T, labels=dict(x='row', y='column', color='missing'))
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr_matrix, zmin=-1, zmax=1, color_continuous_scale='RdBu')
    return fig


def plot_scatter_with_trend(x: pd.Series, y: pd.Series, x_name: str, y_name: str) -> go.Figure:
    df = pd.DataFrame({x_name: x, y_name: y}).dropna()
    fig = px.scatter(df, x=x_name, y=y_name, trendline='ols')
    return fig


def plot_box_grouped(numeric: pd.Series, group: pd.Series, numeric_name: str, group_name: str) -> go.Figure:
    df = pd.DataFrame({numeric_name: numeric, group_name: group}).dropna()
    # ensure group ordering: if numeric-like, order by numeric value
    try:
        gnum = pd.to_numeric(df[group_name], errors='coerce')
        if gnum.notna().sum() / max(1, df.shape[0]) >= 0.9:
            order = [str(v) for v in sorted(gnum.dropna().unique())]
            fig = px.box(df, x=group_name, y=numeric_name, category_orders={group_name: order})
            return fig
    except Exception:
        pass
    fig = px.box(df, x=group_name, y=numeric_name)
    return fig


def plot_stacked_bar_normalized(cat_feature: pd.Series, cat_target: pd.Series, feature_name: str, target_name: str) -> go.Figure:
    ct = pd.crosstab(cat_feature, cat_target)
    prop = ct.div(ct.sum(axis=1), axis=0)
    fig = go.Figure()
    # ensure index ordering for numeric-like categories
    try:
        idx_num = pd.to_numeric(prop.index, errors='coerce')
        if idx_num.notna().sum() / max(1, len(idx_num)) >= 0.9:
            ordered_index = [x for _, x in sorted(zip(idx_num, prop.index))]
        else:
            ordered_index = list(prop.index.astype(str))
    except Exception:
        ordered_index = list(prop.index.astype(str))
    for col in prop.columns:
        fig.add_trace(go.Bar(name=str(col), x=ordered_index, y=prop.reindex(ordered_index).loc[:, col].values))
    fig.update_layout(barmode='stack')
    return fig


def plot_target_over_time_numeric(datetime_col: pd.Series, target: pd.Series) -> go.Figure:
    df = pd.DataFrame({'dt': pd.to_datetime(datetime_col), 'target': target}).dropna()
    if df.empty:
        return go.Figure()
    df = df.set_index('dt').resample('D').mean().dropna()
    fig = px.line(df, x=df.index.astype(str), y='target')
    return fig


def plot_target_over_time_categorical(datetime_col: pd.Series, target: pd.Series) -> go.Figure:
    df = pd.DataFrame({'dt': pd.to_datetime(datetime_col), 'target': target}).dropna()
    if df.empty:
        return go.Figure()
    df = df.set_index('dt').groupby(pd.Grouper(freq='D'))['target'].value_counts(normalize=True).unstack().fillna(0)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=[str(x) for x in df.index], y=df[col], name=str(col), stackgroup='one'))
    return fig


def fig_to_html_div(fig: go.Figure, div_id: str, include_js: bool = False) -> str:
    # Ensure embedded figures have a concrete pixel height to avoid
    # zero-height containers (which cause invisible plots / Plotly sad face).
    try:
        height = fig.layout.height
    except Exception:
        height = None
    if not height:
        fig.update_layout(height=400)
    html = fig.to_html(full_html=False, include_plotlyjs=include_js, div_id=div_id)

    # If the figure contains scatter marker traces, provide a PNG fallback
    # embedded as base64. This helps when interactive renderer fails to show markers.
    try:
        has_scatter_markers = False
        for t in getattr(fig, 'data', []):
            t_type = getattr(t, 'type', None)
            t_mode = getattr(t, 'mode', '') or ''
            if (t_type and 'scatter' in t_type) or ('markers' in t_mode):
                has_scatter_markers = True
                break
        if has_scatter_markers:
            try:
                img_bytes = fig.to_image(format='png')
            except Exception:
                # fallback: render a simple matplotlib scatter using first scatter trace
                try:
                    import matplotlib.pyplot as plt
                    from io import BytesIO
                    # find first scatter trace with x/y
                    tx = None
                    ty = None
                    for t in getattr(fig, 'data', []):
                        t_type = getattr(t, 'type', None)
                        t_mode = getattr(t, 'mode', '') or ''
                        if (t_type and 'scatter' in t_type) or ('markers' in t_mode):
                            tx = list(getattr(t, 'x', []))
                            ty = list(getattr(t, 'y', []))
                            if tx and ty:
                                break
                    if tx is not None and ty is not None and len(tx) and len(ty):
                        fig_mpl = plt.figure(figsize=(6, 4), dpi=150)
                        ax = fig_mpl.add_subplot(1, 1, 1)
                        ax.scatter(tx, ty, s=10, c='#636efa', edgecolors='k', linewidths=0.5, alpha=0.95)
                        ax.set_xlabel(getattr(fig.layout, 'xaxis', {}).get('title', '') if hasattr(fig, 'layout') else '')
                        ax.set_ylabel(getattr(fig.layout, 'yaxis', {}).get('title', '') if hasattr(fig, 'layout') else '')
                        buf = BytesIO()
                        fig_mpl.tight_layout()
                        fig_mpl.savefig(buf, format='png')
                        plt.close(fig_mpl)
                        img_bytes = buf.getvalue()
                    else:
                        raise
                except Exception:
                    raise
            b64 = base64.b64encode(img_bytes).decode('ascii')
            img_tag = f"<div><img src=\"data:image/png;base64,{b64}\" style=\"max-width:100%;height:auto;display:block;margin-top:8px;\"/></div>"
            html = html + img_tag
    except Exception:
        # if image generation fails (no kaleido), ignore and return interactive html
        pass

    return html
