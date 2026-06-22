import pandas as pd
import markdown2
import plotly.io as pio


from src.visualizations import (
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_missing_values,
    plot_correlation_heatmap,
    plot_target_distribution,
    plot_categorical_cardinality,
    generate_variable_summary_table,
    plot_target_correlations,
)

# --------------------------------------------------
# EDA-Only Report Builder
# --------------------------------------------------

import markdown2
import plotly.io as pio

def build_eda_only_report(
    df: pd.DataFrame,
    profile: dict,
    summary_narrative: str,
    selected_target: str | None = None
) -> str:
    """
    Build an EDA-only report.
    Combines Markdown intro with HTML summary table and Plotly visualizations.
    """

    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # --- Intro en Markdown ---
    md = [
        f"# DataInsight AI - Automated EDA Report\n",
        f"## 🔢 Dataset Metrics",
        f"- **Total Rows:** {df.shape[0]:,}",
        f"- **Total Columns:** {df.shape[1]}",
        f"- **Duplicate Records:** {profile['duplicates']:,}",
        f"- **Memory Footprint:** {memory_usage_mb:.2f} MB\n",
        f"## 📊 Statistical Data Overview",
        summary_narrative
    ]
    md_html = markdown2.markdown("\n".join(md))

    # --- CSS global ---
    style_block = """
    <style>
    body { background-color: #f5f5f5; color: #222; font-family: 'Segoe UI', sans-serif; }
    .plotly-graph-div { background-color: #f5f5f5 !important; }
    h1, h2, h3 { color: #0b61a4; }
    </style>
    """

    # --- Variables Summary Table ---
    summary_table = generate_variable_summary_table(df)
    html_parts = [
        style_block,
        md_html,
        "<h2>📋 Variables Summary Table</h2>",
        summary_table.to_html(index=False, classes="table table-striped")
    ]

    # --- TAB 1: Distributions ---
    html_parts.append("<h2>📊 Distributions</h2>")
    fig_num = plot_numerical_distributions(df)
    if fig_num:
        html_parts.append(pio.to_html(fig_num, full_html=False, include_plotlyjs="cdn"))
    fig_cat = plot_categorical_distributions(df)
    if fig_cat:
        html_parts.append(pio.to_html(fig_cat, full_html=False, include_plotlyjs="cdn"))

    # --- TAB 2: Quality & Structure ---
    html_parts.append("<h2>🔍 Quality & Structure</h2>")
    fig_missing = plot_missing_values(df)
    if fig_missing:
        html_parts.append(pio.to_html(fig_missing, full_html=False, include_plotlyjs="cdn"))
    fig_card = plot_categorical_cardinality(df)
    if fig_card:
        html_parts.append(pio.to_html(fig_card, full_html=False, include_plotlyjs="cdn"))

    # --- TAB 3: Data Relations ---
    html_parts.append("<h2>🔗 Data Relations</h2>")
    fig_corr = plot_correlation_heatmap(df)
    if fig_corr:
        html_parts.append(pio.to_html(fig_corr, full_html=False, include_plotlyjs="cdn"))
        
    # --- TAB 4: Target Analysis (Conditional Section) ---
    # Render the target metrics only if a valid string is provided and exists as a dataframe feature
    if selected_target and selected_target in df.columns:
        html_parts.append(f"<h2>🎯 Target Analysis: {selected_target}</h2>")
        
        # Extract and append the isolated target classification/regression distribution chart
        fig_target = plot_target_distribution(df, selected_target)
        if fig_target:
            html_parts.append(pio.to_html(fig_target, full_html=False, include_plotlyjs="cdn"))
            
        # Extract and append the ANOVA/Chi-Square predictive feature power chart
        fig_target_corr = plot_target_correlations(df, selected_target)
        if fig_target_corr:
            html_parts.append(pio.to_html(fig_target_corr, full_html=False, include_plotlyjs="cdn"))
    else:
        # Fallback tracking if the user explicitly chose 'None' or did an unsupervised execution run
        html_parts.append("<h2>🎯 Target Analysis</h2><p><i>No target variable was selected for this analytical execution run.</i></p>")

    # Merge the structural HTML array blocks into a single uniform web document stream
    return "\n".join(html_parts)



# --------------------------------------------------
# AI-Only Report Builder
# --------------------------------------------------

def build_ai_only_report(ai_report: str) -> str:
    """
    Build an AI-only report.
    Assembles OpenAI insights and predictive modeling strategies into HTML.
    """
    style_block = """
    <style>
    body { background-color: #f5f5f5; color: #222; font-family: 'Segoe UI', sans-serif; }
    .plotly-graph-div { background-color: #f5f5f5 !important; }
    h1, h2, h3 { color: #0b61a4; }
    </style>
    """

    md = [
        f"# DataInsight AI - OpenAI Predictive Modeling Strategy\n",
        ai_report
    ]
    md_html = markdown2.markdown("\n".join(md))  # convierte Markdown a HTML

    return "\n".join([style_block, md_html])


# --------------------------------------------------
# Full Executive Report Builder
# --------------------------------------------------

def build_full_report(df: pd.DataFrame, profile: dict, summary_narrative: str, ai_report: str) -> str:
    """
    Build a full report.
    
    Combines both EDA metrics and OpenAI insights into a comprehensive Markdown document.
    Includes dataset metrics, statistical overview, and predictive ML strategies.
    """
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    style_block = """
    <style>
    body { background-color: #f5f5f5; color: #222; font-family: 'Segoe UI', sans-serif; }
    .plotly-graph-div { background-color: #f5f5f5 !important; }
    h1, h2, h3 { color: #0b61a4; }
    </style>
    """

    md = [
        f"# DataInsight AI - Full Executive Analytics Report\n",
        f"## 🔢 Dataset Metrics",
        f"- **Total Rows:** {df.shape[0]:,}",
        f"- **Total Columns:** {df.shape[1]}",
        f"- **Duplicate Records:** {profile['duplicates']:,}",
        f"- **Memory Footprint:** {memory_usage_mb:.2f} MB\n",
        f"## 📊 Statistical Data Overview",
        summary_narrative,
        f"\n## 🧠 OpenAI Insights & ML Strategy",
        ai_report
    ]

    md_html = markdown2.markdown("\n".join(md))

    return "\n".join([style_block, md_html])

# --------------------------------------------------
# HTML Report Export Helper
# --------------------------------------------------

def export_report_to_html(md_content: str) -> str:
    """
    Convert Markdown content into HTML string.
    """
    return markdown2.markdown(md_content)