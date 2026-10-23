import jinja2
import os
import re
from plotly.offline import get_plotlyjs


def _slugify(value) -> str:
    slug = re.sub(r'[^A-Za-z0-9]+', '-', str(value)).strip('-').lower()
    return slug or 'col'


def generate_report(
    df,
    profiles,
    quality_flags,
    univariate_html,
    target_deep_dive_html,
    bivariate_html,
    correlation_html,
    vif_table,
    stats_test_table,
    ml_readiness_table,
    target,
    problem_type,
    output_path,
    template_path=None,
):
    if template_path is None:
        template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report_template.html')

    env = jinja2.Environment(loader=jinja2.FileSystemLoader('/'), autoescape=True)
    env.filters['slugify'] = _slugify
    template = env.get_template(template_path)
    plotlyjs = get_plotlyjs()
    rendered = template.render(
        plotlyjs=plotlyjs,
        df=df,
        profiles=profiles,
        quality_flags=quality_flags,
        univariate_html=univariate_html,
        target_deep_dive_html=target_deep_dive_html,
        bivariate_html=bivariate_html,
        correlation_html=correlation_html,
        vif_table=vif_table.to_html(index=False) if vif_table is not None else None,
        stats_test_table=stats_test_table.to_html(index=False) if stats_test_table is not None else None,
        ml_readiness_table=ml_readiness_table.to_html(index=False) if ml_readiness_table is not None else None,
        target=target,
        problem_type=problem_type,
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)
    return output_path
