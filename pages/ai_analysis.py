"""
AI Analysis
"""

import streamlit as st
from src.profiler import profile_dataset  # Ensure the dataset profiling module is available

# 1. Global navigation button header to safely return back to Home (Always visible)
if st.button("⬅️ Back to Home", type="secondary"):
    st.switch_page("pages/home.py")

st.markdown("# 🧠 OpenAI Insights Report")

# ------------------------------------------------------------------
# CROSS-PAGE ROUTING & VALIDATION PIPELINE
# ------------------------------------------------------------------

# 1. Verify if the baseline DataFrame is correctly retained in the global cache memory
has_df = "df_shared" in st.session_state and st.session_state.df_shared is not None

# 2. Extract and unify the security key payload from the Home button package OR directly from the Sidebar widget
raw_key = st.session_state.get("api_key_shared") or st.session_state.get("user_api_key")
has_api_key = raw_key is not None and bool(str(raw_key).strip())

# 3. Target label choice validation tracks
has_target_shared = "target_shared" in st.session_state
has_target_selection = "target_selection" in st.session_state and st.session_state.target_selection is not None
has_target_choice = has_target_shared or has_target_selection

allow_ai = st.session_state.get("allow_ai", False)

# ------------------------------------------------------------------
# MAIN PIPELINE FLUX: Only executes if all security checkpoints pass
# ------------------------------------------------------------------

if allow_ai and has_df and has_target_shared and has_api_key:
    df = st.session_state.df_shared
    target = st.session_state.get('target_shared')
    api_key = str(raw_key).strip()  # Clean spaces to feed a pristine credentials string to the official SDK
    
    # Initialize the core asynchronous content tracker within the cache if missing
    if "ai_report_output" not in st.session_state:
        st.session_state.ai_report_output = None

    # Performance safeguard: Only trigger OpenAI if a previous report hasn't been generated in this session
    if st.session_state.ai_report_output is None:
        
        with st.status('🧠 Generating expert analysis with AI...', expanded=True) as status:
            try:
                st.write('⚙️ Extracting dataset statistical metadata fields...')
                # 1. Run the structural profiling algorithm against the active file and target context
                profile_data = profile_dataset(df=df, target=target)
                
                st.write('📋 Sampling a lightweight matrix slice (Head JSON)...')
                # 2. Serialize an atomic dataframe subset (3 rows) into string types to prevent structural formatting payload breaks
                sample_data = df.head(3).astype(str).to_dict(orient="records")
                
                st.write('🔌 Connecting to OpenAI Chat Completions API (Model: gpt-4o-mini)...')
                # Lazy-load backend packages and the official exception catchers
                from src.ai_generate_analysis import generate_dataset_analysis
                from openai import AuthenticationError
                
                st.write('🤖 Processing statistical patterns and drafting data science strategy report...')
                # 3. Transmit payload over HTTPS to request the custom markdown analytical file
                reporte_markdown = generate_dataset_analysis(
                    profile=profile_data,
                    sample_rows=sample_data,
                    api_key=api_key
                )
                
                # 4. Safely back up the output text inside the session store to enable costless cross-page retrieval
                st.session_state.ai_report_output = reporte_markdown
                status.update(label='✅ AI Report generated successfully!', state='complete', expanded=False)
                
            except AuthenticationError:
                status.update(label='❌ Authentication Error', state='error', expanded=True)
                st.error(
                    "The provided OpenAI API Key **is not valid** or has expired. "
                    "Please navigate back to Home and double-check your credentials or account credit status."
                )
            except Exception as e:
                status.update(label='❌ Processing Error', state='error', expanded=True)
                st.error(f"An unexpected failure occurred while building your AI report: {str(e)}")

    # ------------------------------------------------------------------
    # Full-Width Markdown Render Panel
    # ------------------------------------------------------------------
    # Render content natively on the screen instantly if it already exists in the session store
    if st.session_state.ai_report_output is not None:
        with st.container(border=True):
            # Streamlit converts the AI markdown structure, bold tokens, and Scikit-Learn blocks flawlessly
            st.markdown(st.session_state.ai_report_output)

# ------------------------------------------------------------------
# DIRECT ACCESS CONTROL CHECKS (Sidebar / URL fallback restriction guards)
# ------------------------------------------------------------------
else:
    # Scenario 1: Global database state is completely missing (No CSV loaded)
    if not has_df:
        st.warning(
            "⚠️ No data has been detected in this session. Please return to the Home page "
            "to upload your CSV file."
        )
        
    # Scenario 2: File is present but the user bypassed picking a validation path track (Target is unassigned)
    elif st.session_state.get('target_selection') is None and not has_target_shared:
        st.warning(
            "⚠️ No target variable has been selected. Please return to the Home page "
            "and pick a target option (or select 'None')."
        )
        
    # Scenario 3: File and Target context are valid, but the Sidebar password block was left completely empty
    elif not has_api_key:
        st.info(
            "🔑 **OpenAI API Key Missing:** Please return to the Home page, input a valid access token "
            "in the left sidebar configuration panel, and click the **'✨ Analyze with AI'** button.",
            icon="ℹ️"
        )
        
    # Scenario 4: All criteria met, but user attempted to jump routes via side-navigation without pushing the action button
    else:
        st.info(
            "You have loaded a dataset, selected a target, and entered your API Key, but you must open this page "
            "by clicking the **'Analyze with AI'** button on the Home page to trigger the generation process.\n\n"
            "Go back to Home and press **'Analyze with AI'** to start the analysis."
        )



