import pandas as pd
import markdown2

# --------------------------------------------------
# EDA-Only Report Builder
# --------------------------------------------------

def build_eda_only_report(df: pd.DataFrame, profile: dict, summary_narrative: str) -> str:
    """
    Build an EDA-only report.
    
    Assembles Automated EDA metrics and summaries into a Markdown string.
    Includes dataset dimensions, duplicates, memory usage, and statistical overview.
    """
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
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
    return "\n".join(md)

# --------------------------------------------------
# AI-Only Report Builder
# --------------------------------------------------

def build_ai_only_report(ai_report: str) -> str:
    """
    Build an AI-only report.
    
    Assembles OpenAI insights and predictive modeling strategies into a Markdown string.
    """
    md = [
        f"# DataInsight AI - OpenAI Predictive Modeling Strategy\n",
        ai_report
    ]
    return "\n".join(md)

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

    md = [
        f"# DataInsight AI - Full Executive Analytics Report\n",
        f"## 🔢 Dataset Metrics",
        f"- **Total Rows:** {df.shape[0]:,}",
        f"- **Total Columns:** {df.shape[1]}",
        f"- **Duplicate Records:** {profile['duplicates']:,}",
        f"- **Memory Footprint:** {memory_usage_mb:.2f} MB\n",
        f"## 📊 Statistical Data Overview",
        summary_narrative,
        f"\n## 🧠 OpenAI Deep Insights & ML Strategy",
        ai_report
    ]
    return "\n".join(md)

# --------------------------------------------------
# HTML Report Export Helper
# --------------------------------------------------

def export_report_to_html(md_content: str) -> str:
    """
    Convert Markdown content into HTML string.
    """
    return markdown2.markdown(md_content)