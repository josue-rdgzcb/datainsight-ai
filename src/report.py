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

def build_eda_only_report(
    df: pd.DataFrame,
    profile: dict,
    summary_narrative: str,
    selected_target: str | None = None
) -> str:
    """
    Build an EDA-only report wrapped in a valid structured HTML skeleton.
    Combines a Markdown introduction with an HTML summary table and interactive Plotly visualizations.
    """

    # Compute the total memory footprint of the DataFrame in Megabytes (MB).
    # This gives the user an idea of dataset size and resource usage.
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # --- Markdown Intro Section ---
    # This block provides dataset-level metrics and a narrative summary.
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

    # --- Global CSS Styling ---
    # Provides consistent styling for the entire report: typography, layout,
    # Plotly graphs, and Pandas summary tables.
    style_block = """
    <style>
    body { background-color: #f5f5f5; color: #222; font-family: 'Segoe UI', -apple-system, sans-serif; padding: 30px; line-height: 1.6; max-width: 1200px; margin: 0 auto; }
    .plotly-graph-div { background-color: #f5f5f5 !important; margin-bottom: 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0b61a4; margin-top: 25px; font-weight: 700; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
    ul { padding-left: 20px; }
    li { margin-bottom: 6px; }
    
    /* Elegant Pandas Summary Table Styling */
    table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 15px; margin-bottom: 30px; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; }
    th { background-color: #0b61a4; color: #ffffff; padding: 12px 16px; font-weight: 600; text-align: left; font-size: 14px; }
    td { padding: 12px 16px; border-bottom: 1px solid #edf2f7; color: #4a5568; font-size: 14px; }
    tr:last-child td { border-bottom: none; }
    tr:nth-child(even) { background-color: #f8fafc; }
    </style>
    """

    # --- Variables Summary Table ---
    # Generate a tabular overview of variables (numeric and categorical).
    summary_table = generate_variable_summary_table(df)
    
    # Initialize the HTML block collector that will hold all report sections.
    body_parts = [
        md_html,
        "<h2>📋 Variables Summary Table</h2>",
        summary_table.to_html(index=False, classes="table")  # Apply custom CSS styling
    ]

    # --- TAB 1: Distributions ---
    # Numerical and categorical distributions provide insight into variable spread and balance.
    body_parts.append("<h2>📊 Distributions</h2>")
    fig_num = plot_numerical_distributions(df)
    if fig_num:
        body_parts.append(pio.to_html(fig_num, full_html=False, include_plotlyjs="cdn"))
    fig_cat = plot_categorical_distributions(df)
    if fig_cat:
        body_parts.append(pio.to_html(fig_cat, full_html=False, include_plotlyjs="cdn"))

    # --- TAB 2: Quality & Structure ---
    # Missing values and categorical cardinality highlight data quality issues.
    body_parts.append("<h2>🔍 Quality & Structure</h2>")
    fig_missing = plot_missing_values(df)
    if fig_missing:
        body_parts.append(pio.to_html(fig_missing, full_html=False, include_plotlyjs="cdn"))
    fig_card = plot_categorical_cardinality(df)
    if fig_card:
        body_parts.append(pio.to_html(fig_card, full_html=False, include_plotlyjs="cdn"))

    # --- TAB 3: Data Relations ---
    # Correlation heatmap shows relationships between numerical variables.
    body_parts.append("<h2>🔗 Data Relations</h2>")
    fig_corr = plot_correlation_heatmap(df)
    if fig_corr:
        body_parts.append(pio.to_html(fig_corr, full_html=False, include_plotlyjs="cdn"))
        
    # --- TAB 4: Target Analysis (Conditional Section) ---
    # If a target variable is selected, provide distribution and correlation analysis.
    if selected_target and selected_target in df.columns:
        body_parts.append(f"<h2>🎯 Target Analysis: {selected_target}</h2>")
        
        fig_target = plot_target_distribution(df, selected_target)
        if fig_target:
            body_parts.append(pio.to_html(fig_target, full_html=False, include_plotlyjs="cdn"))
            
        fig_target_corr = plot_target_correlations(df, selected_target)
        if fig_target_corr:
            body_parts.append(pio.to_html(fig_target_corr, full_html=False, include_plotlyjs="cdn"))
    else:
        body_parts.append("<h2>🎯 Target Analysis</h2><p><i>No target variable was selected for this analytical execution run.</i></p>")

    # --- Assemble Report Body ---
    inner_body_html = "\n".join(body_parts)

    # --- Final HTML Skeleton ---
    # Wrap everything inside a complete HTML document with head, body, and metadata.
    full_html_document = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataInsight AI - Automated EDA Report</title>
    {style_block}
</head>
<body>
    {inner_body_html}
</body>
</html>
"""

    return full_html_document



# --------------------------------------------------
# AI-Only Report Builder
# --------------------------------------------------

def build_ai_only_report(ai_report: str) -> str:
    """
    Build an AI-only report.
    Assembles OpenAI insights and predictive modeling strategies into a fully structured HTML document.
    """

    # 1. Define the global CSS style block
    # This ensures consistent styling across the entire report:
    # - Background and text colors
    # - Typography and spacing
    # - Headings, lists, and code blocks
    # - Plotly graph containers
    style_block = """
    <style>
    body { background-color: #f5f5f5; color: #222; font-family: 'Segoe UI', sans-serif; padding: 20px; line-height: 1.6; }
    .plotly-graph-div { background-color: #f5f5f5 !important; }
    h1, h2, h3 { color: #0b61a4; margin-top: 20px; }
    ul, ol { padding-left: 20px; }
    li { margin-bottom: 5px; }
    pre { background-color: #e2e8f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
    code { font-family: 'Consolas', 'Courier New', monospace; font-size: 14px; }
    </style>
    """

    # 2. Compile the report body from Markdown into pure HTML
    # Markdown is used for readability and structure, then converted to HTML
    # so it can be embedded directly into the final document.
    md_content = [
        "# DataInsight AI - OpenAI Predictive Modeling Strategy\n",
        ai_report
    ]
    md_html = markdown2.markdown("\n".join(md_content))

    # 3. Assemble the official HTML page structure
    # Wraps everything into a complete HTML skeleton with:
    # - DOCTYPE declaration
    # - <head> metadata (charset, viewport, title, CSS)
    # - <body> containing the converted Markdown content
    full_html_document = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataInsight AI - Report</title>
    {style_block}
</head>
<body>
    {md_html}
</body>
</html>
"""

    # Return the fully assembled HTML document string
    return full_html_document


# --------------------------------------------------
# Full Executive Report Builder
# --------------------------------------------------

def build_full_report(
    df: pd.DataFrame,
    profile: dict,
    summary_narrative: str,
    ai_report: str,
    selected_target: str | None = None
) -> str:
    """
    Build a full executive report combining EDA and AI insights.
    Merges both analytical frameworks into a single unified HTML5 document.
    """

    # Calculate the total DataFrame memory footprint in Megabytes (MB).
    # This provides context about dataset size and resource usage.
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # 1. --- PART A: Compile EDA Introduction and Metrics ---
    # Markdown block summarizing dataset metrics and narrative overview.
    eda_md = [
        f"## 🔢 Dataset Metrics",
        f"- **Total Rows:** {df.shape[0]:,}",
        f"- **Total Columns:** {df.shape[1]}",
        f"- **Duplicate Records:** {profile['duplicates']:,}",
        f"- **Memory Footprint:** {memory_usage_mb:.2f} MB\n",
        f"## 📊 Statistical Data Overview",
        summary_narrative
    ]
    eda_md_html = markdown2.markdown("\n".join(eda_md))

    # Generate structured variable summary table (numeric + categorical).
    summary_table = generate_variable_summary_table(df)
    
    # Initialize unified body collector for HTML sections.
    html_body_parts = [
        "<h1>📊 DataInsight AI - Full Executive Analytics Report</h1>",
        "<hr style='border: 0; border-top: 2px solid #0b61a4; margin-bottom: 30px;'>",
        "<h2>🔍 Section 1: Exploratory Data Analysis (EDA)</h2>",
        eda_md_html,
        "<h3>📋 Variables Summary Table</h3>",
        summary_table.to_html(index=False, classes="table")
    ]

    # Sequentially inject EDA visualizations.
    html_body_parts.append("<h3>📊 Distributions</h3>")
    fig_num = plot_numerical_distributions(df)
    if fig_num:
        html_body_parts.append(pio.to_html(fig_num, full_html=False, include_plotlyjs="cdn"))
    fig_cat = plot_categorical_distributions(df)
    if fig_cat:
        html_body_parts.append(pio.to_html(fig_cat, full_html=False, include_plotlyjs="cdn"))

    html_body_parts.append("<h3>🔍 Quality & Structure</h3>")
    fig_missing = plot_missing_values(df)
    if fig_missing:
        html_body_parts.append(pio.to_html(fig_missing, full_html=False, include_plotlyjs="cdn"))
    fig_card = plot_categorical_cardinality(df)
    if fig_card:
        html_body_parts.append(pio.to_html(fig_card, full_html=False, include_plotlyjs="cdn"))

    html_body_parts.append("<h3>🔗 Data Relations</h3>")
    fig_corr = plot_correlation_heatmap(df)
    if fig_corr:
        html_body_parts.append(pio.to_html(fig_corr, full_html=False, include_plotlyjs="cdn"))
        
    # Conditional target analysis if a target variable is provided.
    if selected_target and selected_target in df.columns:
        html_body_parts.append(f"<h3>🎯 Target Analysis: {selected_target}</h3>")
        fig_target = plot_target_distribution(df, selected_target)
        if fig_target:
            html_body_parts.append(pio.to_html(fig_target, full_html=False, include_plotlyjs="cdn"))
        fig_target_corr = plot_target_correlations(df, selected_target)
        if fig_target_corr:
            html_body_parts.append(pio.to_html(fig_target_corr, full_html=False, include_plotlyjs="cdn"))
    else:
        html_body_parts.append("<h3>🎯 Target Analysis</h3><p><i>No target variable was selected for this analytical execution run.</i></p>")

    # 2. --- PART B: Compile AI Insights ---
    # Append AI-generated recommendations and strategies.
    html_body_parts.append("<hr style='border: 0; border-top: 2px solid #0b61a4; margin-top: 50px; margin-bottom: 30px;'>")
    html_body_parts.append("<h2>🤖 Section 2: OpenAI Predictive Modeling Strategy</h2>")
    
    # Convert AI recommendations from Markdown to native HTML components.
    ai_md_html = markdown2.markdown(ai_report)
    html_body_parts.append(ai_md_html)

    # 3. --- PART C: Define Unified Premium CSS Styling ---
    # Provides consistent design across headings, tables, code blocks, and graphs.
    style_block = """
    <style>
    body { background-color: #f5f5f5; color: #222; font-family: 'Segoe UI', -apple-system, sans-serif; padding: 40px; line-height: 1.6; max-width: 1200px; margin: 0 auto; }
    .plotly-graph-div { background-color: #f5f5f5 !important; margin-bottom: 25px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    h1 { color: #0f172a; text-align: center; font-size: 36px; font-weight: 800; margin-bottom: 10px; }
    h2 { color: #0b61a4; margin-top: 35px; font-size: 26px; font-weight: 700; border-bottom: 2px solid #0b61a4; padding-bottom: 6px; }
    h3 { color: #1e293b; margin-top: 25px; font-size: 20px; font-weight: 600; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
    ul, ol { padding-left: 20px; }
    li { margin-bottom: 6px; }
    
    /* Preprocessing Code Blocks Styles */
    pre { background-color: #e2e8f0; padding: 14px; border-radius: 6px; overflow-x: auto; border: 1px solid #cbd5e1; margin-bottom: 20px; }
    code { font-family: 'Consolas', 'Courier New', monospace; font-size: 14px; color: #0f172a; }
    
    /* Unified Interactive Table Styling */
    table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 15px; margin-bottom: 30px; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; }
    th { background-color: #0b61a4; color: #ffffff; padding: 12px 16px; font-weight: 600; text-align: left; font-size: 14px; }
    td { padding: 12px 16px; border-bottom: 1px solid #edf2f7; color: #4a5568; font-size: 14px; }
    tr:last-child td { border-bottom: none; }
    tr:nth-child(even) { background-color: #f8fafc; }
    </style>
    """

    # Merge all body components into a single string.
    master_body_content = "\n".join(html_body_parts)

    # 4. --- PART D: Build Final HTML Skeleton ---
    # Wrap everything into a complete HTML5 document with head, body, and metadata.
    full_html_document = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataInsight AI - Full Executive Analytics Report</title>
    {style_block}
</head>
<body>
    {master_body_content}
</body>
</html>
"""

    return full_html_document


# --------------------------------------------------
# HTML Report Export Helper
# --------------------------------------------------

def export_report_to_html(md_content: str) -> str:
    """
    Convert Markdown content into an HTML string.
    Useful for exporting AI or EDA narrative sections into HTML format.
    """
    return markdown2.markdown(md_content)
