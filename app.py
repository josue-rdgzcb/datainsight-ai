"""
Main Application Entry Point - Orchestration Router
"""

import streamlit as st

# Global window configuration (inherited by all pages across the multipage app architecture)
st.set_page_config(
    page_title="DataInsight AI", 
    page_icon="static/ai_assistant.png", 
    layout="wide"
)

# Page Declarations: Pointing to the isolated subpage module files
home_page = st.Page(
    "pages/home.py", 
    title="Data Explorer & Visualizations", 
    icon="📊", 
    default=True
)

ai_page = st.Page(
    "pages/ai_analysis.py", 
    title="AI Insights Report", 
    icon="🧠"
)

reports_page = st.Page(
    "pages/reports.py",
    title="Download Reports",
    icon="📥"
)


# Instantiate native Streamlit navigation controls matching the sidebar directory structure
pg = st.navigation([home_page, ai_page, reports_page])

# Execute and dynamically render the subpage routing according to the user's focus
pg.run()

# --------------------------------------------------
# Global Footer Signature (Sidebar Bottom)
# --------------------------------------------------
with st.sidebar:
    st.markdown("---") 
    st.markdown(
        """
        <div style="text-align: center; color: #64748B; font-size: 13px; font-weight: 500;">
            Developed by Josué Rodríguez
            <div style="margin-top: 6px;">
                <a href="https://github.com/josue-rdgzcb" target="_blank" style="color: #0b61a4; font-weight: 700; text-decoration: underline;">
                    🔗 GitHub
                </a>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )




