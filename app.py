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

# 1. Page Declarations: Pointing to the isolated subpage module files
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

# 2. Instantiate native Streamlit navigation controls matching the sidebar directory structure
pg = st.navigation([home_page, ai_page])

# 3. Execute and dynamically render the subpage routing according to the user's focus
pg.run()



