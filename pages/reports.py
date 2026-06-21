"""
Download Reports Page
"""

import streamlit as st
import pandas as pd
from src.report import (
    build_eda_only_report,
    build_ai_only_report,
    build_full_report,
    export_report_to_html
)

# Global navigation button
if st.button("⬅️ Back to Home", type="secondary"):
    st.switch_page("pages/home.py")

st.markdown("# 📥 Executive Document Export Center")
st.write("Download tailored offline Markdown (.md) documents featuring your custom data summaries and AI pipelines.")

# Check state matrices from previous workflows
has_df = "df_shared" in st.session_state and st.session_state.df_shared is not None
has_eda = "target_selection_persistent" in st.session_state and st.session_state.target_selection_persistent is not None
has_ai = "ai_report_output" in st.session_state and st.session_state.ai_report_output is not None

# Basic safety block: If no file is loaded, freeze everything
if not has_df:
    st.warning("⚠️ No dataset detected. Please go back to the Home page and upload a CSV file to enable exports.")
else:
    df = st.session_state.df_shared
    file_prefix = st.session_state.get("loaded_file_name", "dataset").split('.')[0]
    
    # We attempt to safely pull the cached statistical profile and overview description
    # These are stored in home.py downstream workflow natively
    profile_data = st.session_state.get("profile_data_cache")
    summary_data = st.session_state.get("summary_data_cache")
    ai_report_data = st.session_state.get("ai_report_output")

    st.markdown("---")
    
    # ------------------------------------------------------------------
    # BUTTON 1: Automated EDA Only Export (HTML)
    # ------------------------------------------------------------------
    with st.container(border=True):
        col_btn1, col_txt1 = st.columns([1.2, 2.0], vertical_alignment="center")
        with col_btn1:
            eda_disabled = not (has_eda and profile_data and summary_data)
            
            if not eda_disabled:
                eda_md = build_eda_only_report(df, profile_data, summary_data)
                eda_html = export_report_to_html(eda_md)

                st.download_button(
                    label="📥 Download Automated EDA (.html)",
                    data=eda_html,
                    file_name=f"eda_report_{file_prefix}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_eda_html_btn"
                )
            else:
                st.button("📥 Download Automated EDA (.html)", disabled=True, use_container_width=True, key="disabled_eda_html_btn")
                
        with col_txt1:
            if eda_disabled:
                st.caption("🔒 *Locked: Select a target variable or 'None' on the Home page to unlock the automated EDA download.*")
            else:
                st.info("✅ **Ready:** Contains row/column metrics, data profiling statistics, and the text data overview.", icon="📊")



    # ------------------------------------------------------------------
    # BUTTON 2: OpenAI Analysis Only Export (HTML)
    # ------------------------------------------------------------------
    with st.container(border=True):
        col_btn2, col_txt2 = st.columns([1.2, 2.0], vertical_alignment="center")
        with col_btn2:
            # Active ONLY if OpenAI successfully generated the report text block
            ai_disabled = not has_ai
            
            if not ai_disabled:
                ai_md = build_ai_only_report(ai_report_data)
                ai_html = export_report_to_html(ai_md)  # Convert Markdown → HTML

                st.download_button(
                    label="📥 Download AI Insights Only (.html)",
                    data=ai_html,
                    file_name=f"ai_insights_{file_prefix}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_ai_html_btn"
                )
            else:
                st.button("📥 Download AI Insights Only (.html)", disabled=True, use_container_width=True, key="disabled_ai_html_btn")
                
        with col_txt2:
            if ai_disabled:
                st.caption("🔒 *Locked: Provide your OpenAI API Key and trigger 'Analyze with AI' to unlock this strategic report.*")
            else:
                st.info("✅ **Ready:** Contains custom data-quality alerts, feature engineering suggestions, and Scikit-Learn code scripts.", icon="🧠")


    # ------------------------------------------------------------------
    # BUTTON 3: Full Executive Report Export (EDA + AI) - HTML
    # ------------------------------------------------------------------
    with st.container(border=True):
        col_btn3, col_txt3 = st.columns([1.2, 2.0], vertical_alignment="center")
        with col_btn3:
            # Active ONLY if BOTH workflows are fully completed in the memory track
            full_disabled = not (has_eda and has_ai and profile_data and summary_data)
            
            if not full_disabled:
                full_md = build_full_report(df, profile_data, summary_data, ai_report_data)
                full_html = export_report_to_html(full_md)  # Convert Markdown → HTML

                st.download_button(
                    label="📥 Download Full Executive Report (.html)",
                    data=full_html,
                    file_name=f"full_executive_report_{file_prefix}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_full_html_btn"
                )
            else:
                st.button("📥 Download Full Executive Report (.html)", disabled=True, use_container_width=True, key="disabled_full_html_btn")
                
        with col_txt3:
            if full_disabled:
                st.caption("🔒 *Locked: Requires both a generated Automated EDA and a completed OpenAI execution session.*")
            else:
                st.success("✨ **Fully Unlocked:** Merges both frameworks into one unified master analytical document.", icon="🏆")

