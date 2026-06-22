import pandas as pd
import markdown2
import plotly.io as pio
import numpy as np


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
    Combines Markdown intro with an HTML summary table and interactive Plotly visualizations.
    """
    # Create a lightweight copy of the dataframe to optimize numeric precision
    # This prevents long float strings (e.g., 45.123456789) from bloating the HTML table text size
    df_table_opt = df.copy()
    float_cols = df_table_opt.select_dtypes(include=['float32', 'float64']).columns
    df_table_opt[float_cols] = df_table_opt[float_cols].round(2)

    # Calculate full DataFrame memory footprint in Megabytes (MB)
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # --- Markdown Intro Section ---
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

    # --- Global Premium CSS Styling ---
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
    # Generated using the decimal-optimized dataframe to save massive amounts of string text
    summary_table = generate_variable_summary_table(df_table_opt)
    
    # Initialize the internal HTML components storage array
    body_parts = [
        md_html,
        "<h2>📋 Variables Summary Table</h2>",
        summary_table.to_html(index=False, classes="table") # Pass clean class definitions for the custom CSS engine
    ]

    # --- Performance Optimization Parameter Configuration ---
    # Setting include_plotlyjs=False prevents injecting the heavy 3MB core engine into each individual chart.
    plotly_html_kwargs = {"full_html": False, "include_plotlyjs": False}

    def optimize_and_to_html(fig) -> str:
        """
        Helper function to strip down hidden hover text data arrays from the 
        figure clone right before exporting to HTML to minimize the final payload.
        """
        if not fig:
            return ""
        # Create a shallow copy so we do not mutate the original charts shown in the UI
        fig_copy = pio.from_json(pio.to_json(fig))
        
        # Strip down heavy unneeded hover template text data arrays that bloat JSON size
        fig_copy.update_traces(hoverinfo="x+y", hovertext=None)
        if hasattr(fig_copy.data, '__iter__'):
            for trace in fig_copy.data:
                if 'hoverinfo' in trace:
                    trace.hoverinfo = "x+y"
                if 'text' in trace and isinstance(trace.text, (list, np.ndarray)):
                    # Remove custom array-based hover descriptions that scale with dataframe rows
                    trace.text = None
        
        return pio.to_html(fig_copy, **plotly_html_kwargs)

    # --- TAB 1: Distributions ---
    body_parts.append("<h2>📊 Distributions</h2>")
    fig_num = plot_numerical_distributions(df)
    if fig_num:
        body_parts.append(optimize_and_to_html(fig_num))
    fig_cat = plot_categorical_distributions(df)
    if fig_cat:
        body_parts.append(optimize_and_to_html(fig_cat))

    # --- TAB 2: Quality & Structure ---
    body_parts.append("<h2>🔍 Quality & Structure</h2>")
    fig_missing = plot_missing_values(df)
    if fig_missing:
        body_parts.append(optimize_and_to_html(fig_missing))
    fig_card = plot_categorical_cardinality(df)
    if fig_card:
        body_parts.append(optimize_and_to_html(fig_card))

    # --- TAB 3: Data Relations ---
    body_parts.append("<h2>🔗 Data Relations</h2>")
    fig_corr = plot_correlation_heatmap(df)
    if fig_corr:
        body_parts.append(optimize_and_to_html(fig_corr))
        
    # --- TAB 4: Target Analysis (Conditional Section) ---
    if selected_target and selected_target in df.columns:
        body_parts.append(f"<h2>🎯 Target Analysis: {selected_target}</h2>")
        
        fig_target = plot_target_distribution(df, selected_target)
        if fig_target:
            body_parts.append(optimize_and_to_html(fig_target))
            
        fig_target_corr = plot_target_correlations(df, selected_target)
        if fig_target_corr:
            body_parts.append(optimize_and_to_html(fig_target_corr))
    else:
        body_parts.append("<h2>🎯 Target Analysis</h2><p><i>No target variable was selected for this analytical execution run.</i></p>")

    # Unify all individual HTML parts into a single text body blocks layout
    inner_body_html = "\n".join(body_parts)

    # Wrap up all elements synchronously inside the official standardized web frame skeleton.
    full_html_document = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataInsight AI - Automated EDA Report</title>
    <script src="https://plot.ly"></script>
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

    # 1. Definimos el bloque de estilos CSS limpio
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

    # 2. Compilamos el cuerpo del reporte desde Markdown a HTML puro
    md_content = [
        "# DataInsight AI - OpenAI Predictive Modeling Strategy\n",
        ai_report
    ]
    md_html = markdown2.markdown("\n".join(md_content))

    # 3. Ensamblamos la estructura oficial de una página web autoejecutable
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
    Merges both analytical frameworks into a single unified HTML5 document framework.
    """
    # Calculate full DataFrame memory footprint in Megabytes (MB)
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # 1. --- PARTE A: Compilamos la Introducción y Métricas del EDA ---
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

    # Generamos la tabla estructurada de variables
    summary_table = generate_variable_summary_table(df)
    
    # Inicializamos la recolección del cuerpo unificado
    html_body_parts = [
        "<h1>📊 DataInsight AI - Full Executive Analytics Report</h1>",
        "<hr style='border: 0; border-top: 2px solid #0b61a4; margin-bottom: 30px;'>",
        "<h2>🔍 Section 1: Exploratory Data Analysis (EDA)</h2>",
        eda_md_html,
        "<h3>📋 Variables Summary Table</h3>",
        summary_table.to_html(index=False, classes="table")
    ]

    # Inyectamos de manera secuencial los gráficos del EDA
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
        
    # Agregamos la analítica condicional del target si aplica
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

    # 2. --- PARTE B: Compilamos las Respuestas de la Inteligencia Artificial ---
    html_body_parts.append("<hr style='border: 0; border-top: 2px solid #0b61a4; margin-top: 50px; margin-bottom: 30px;'>")
    html_body_parts.append("<h2>🤖 Section 2: OpenAI Predictive Modeling Strategy</h2>")
    
    # Traducimos las recomendaciones de la IA de Markdown a componentes HTML nativos
    ai_md_html = markdown2.markdown(ai_report)
    html_body_parts.append(ai_md_html)

    # 3. --- PARTE C: Definimos el CSS Unificado Premium de la Suite ---
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

    # Unificamos la lista completa de componentes del cuerpo
    master_body_content = "\n".join(html_body_parts)

    # 4. --- PARTE D: Construimos el esqueleto maestro final de internet ---
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
    Convert Markdown content into HTML string.
    """
    return markdown2.markdown(md_content)