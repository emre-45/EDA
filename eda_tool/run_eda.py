#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from eda_tool.eda_core.data_loader import load_csv, validate_target
from eda_tool.eda_core.type_inference import profile_columns, infer_problem_type
from eda_tool.eda_core.data_quality import build_quality_report
from eda_tool.eda_core.plotting import fig_to_html_div
from eda_tool.eda_core.report import generate_report
from eda_tool.eda_core import plotting
from eda_tool.eda_core import bivariate, correlation, stats_tests, ml_readiness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--target', '-t', required=True)
    parser.add_argument('--output', '-o')
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--sample-threshold', type=int)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s [%(levelname)s] %(message)s')

    try:
        df = load_csv(args.input)
    except FileNotFoundError:
        logging.error(f"Error: input file not found at {args.input}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

    try:
        validate_target(df, args.target)
    except ValueError as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

    profiles = profile_columns(df)
    target_profile = next((p for p in profiles if p.name == args.target), None)
    problem_type = infer_problem_type(target_profile)
    quality_flags = build_quality_report(df, profiles, args.target, problem_type)

    # Univariate plots
    univariate_html = {}
    for p in profiles:
        if p.inferred_type in ('id_like', 'constant', 'free_text'):
            univariate_html[p.name] = 'Plot unavailable for this column'
            continue
        try:
            func = None
            if p.inferred_type.startswith('numerical'):
                fig = plotting.plot_histogram_box(df[p.name], p.name)
            elif p.inferred_type == 'boolean':
                fig = plotting.plot_bar_boolean(df[p.name], p.name)
            elif p.inferred_type.startswith('categorical'):
                fig = plotting.plot_bar_categorical(df[p.name], p.name)
            elif p.inferred_type == 'datetime':
                fig = plotting.plot_datetime_trend(df[p.name], p.name)
            else:
                fig = None
            if fig is None:
                univariate_html[p.name] = 'Plot unavailable for this column'
            else:
                univariate_html[p.name] = fig_to_html_div(fig, div_id=f"univariate-{p.name}")
        except Exception:
            logging.warning(f"Plot unavailable for column {p.name}")
            univariate_html[p.name] = 'Plot unavailable for this column'

    # Target deep dive
    try:
        target_fig = None
        if target_profile.inferred_type.startswith('numerical'):
            target_fig = plotting.plot_histogram_box(df[args.target], args.target)
        elif target_profile.inferred_type.startswith('categorical') or target_profile.inferred_type=='boolean':
            target_fig = plotting.plot_bar_categorical(df[args.target], args.target)
        elif target_profile.inferred_type=='datetime':
            target_fig = plotting.plot_datetime_trend(df[args.target], args.target)
        target_deep_dive_html = fig_to_html_div(target_fig, div_id='target') if target_fig is not None else 'Plot unavailable for target'
    except Exception:
        logging.warning('Target plot unavailable')
        target_deep_dive_html = 'Plot unavailable for target'

    # Bivariate
    ranked = bivariate.rank_features_by_association(df, profiles, args.target, problem_type)
    bivariate_html = []
    for col, score in ranked:
        try:
            p = next(p for p in profiles if p.name == col)
            res = bivariate.compute_association(df, col, args.target, p.inferred_type, target_profile.inferred_type, problem_type)
            fig = res.get('figure')
            html = fig_to_html_div(fig, div_id=f'biv-{col}') if fig is not None else 'Plot unavailable for this column'
            bivariate_html.append((col, res.get('metric_name'), res.get('metric_value'), html))
        except Exception:
            logging.warning(f'Bivariate plot unavailable for {col}')
            bivariate_html.append((col, '', None, 'Plot unavailable for this column'))

    # Correlation & VIF
    numeric_cols = [p.name for p in profiles if p.inferred_type.startswith('numerical')]
    corr_matrix = correlation.compute_correlation_matrix(df, numeric_cols) if numeric_cols else None
    high_corr_pairs = correlation.flag_high_correlation_pairs(corr_matrix) if corr_matrix is not None else []
    vif_table = correlation.compute_vif(df, numeric_cols) if numeric_cols else None
    corr_html = plotting.plot_correlation_heatmap(corr_matrix) if corr_matrix is not None else None
    corr_html = fig_to_html_div(corr_html, 'corr') if corr_html is not None else 'No correlation'

    stats_table = stats_tests.run_all_applicable_tests(df, profiles, args.target, problem_type)
    ml_table = ml_readiness.build_ml_readiness_table(profiles, quality_flags, high_corr_pairs)

    output_path = args.output or (os.path.splitext(os.path.basename(args.input))[0] + '_eda_report.html')
    output_path = os.path.abspath(output_path)

    generate_report(df, profiles, quality_flags, univariate_html, target_deep_dive_html, bivariate_html, corr_html, vif_table, stats_table, ml_table, args.target, problem_type, output_path, template_path=os.path.join(os.path.dirname(__file__), 'templates', 'report_template.html'))

    logging.info(f"Report written to {output_path}")
    print(f"Report written to {output_path}")
    sys.exit(0)


if __name__ == '__main__':
    main()
