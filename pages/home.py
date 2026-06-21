"""
Home

"""

import streamlit as st
import pandas as pd

from src.loader import load_csv
from src.profiler import profile_dataset, generate_data_summary

from src.visualizations import (
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_missing_values,
    plot_correlation_heatmap,
    plot_target_distribution,
    plot_categorical_cardinality,
    generate_variable_summary_table,
    plot_target_correlations,
    plot_target_correlations_2,
    plot_target_mutual_information
)

# --------------------------------------------------
# Session State Initialization (Centralizado al inicio)
# --------------------------------------------------
if "df_shared" not in st.session_state:
    st.session_state.df_shared = None

if "target_selection" not in st.session_state:
    st.session_state.target_selection = None

if "target_selection_persistent" not in st.session_state:
    st.session_state.target_selection_persistent = None

if "allow_ai" not in st.session_state:
    st.session_state.allow_ai = False

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
        <img src="app/static/ai_assistant.png" width="90" style="display: block; object-fit: contain;">
        <h1 style="margin: 0; font-size: 42px; font-weight: 800; color: #0F172A; line-height: 1;">DataInsight AI</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

st.write("Upload a CSV dataset to perform an **automated exploratory analysis** and generate deep **predictive engineering insights powered by AI**.")

# --------------------------------------------------
# Main layout - Top Section
# --------------------------------------------------
with st.sidebar:
    st.markdown("##### AI Settings")
    user_api_key = st.text_input(
        "OpenAI API Key:",
        type="password",  # Masks input characters as dots for data security
        value=st.session_state.user_api_key,
        placeholder="sk-proj-...",
        help="Your key is processed locally within this session's memory and is never shared with external servers."
    )
    # If the user modifies the key, persist it immediately in the session state and refresh
    if user_api_key != st.session_state.user_api_key:
        st.session_state.user_api_key = user_api_key
        st.rerun()


col_upload, col_action = st.columns([1.5, 1.5], vertical_alignment="bottom")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)", 
        type=["csv"]
    )

    dataset_from_session = False
    if uploaded_file is None and st.session_state.df_shared is not None:
        uploaded_file = "__from_session__"
        dataset_from_session = True

with col_action:
    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            height: 68px !important;
            background-color: #EFF6FF !important; 
            border: 1px solid #DBEAFE !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #DBEAFE !important; 
            border-color: #DBEAFE !important;
        }
        div.stButton > button[kind="primary"] p {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    ai_disabled = (
        (uploaded_file is None) 
        or (st.session_state.target_selection_persistent is None)
        or (not st.session_state.user_api_key.strip())
    )
    
    ai_clicked = st.button(
        "✨ Analyze with AI", 
        type="primary", 
        use_container_width=True,
        disabled=ai_disabled
    )

# --------------------------------------------------
# Main workflow
# --------------------------------------------------
if uploaded_file is not None:

    # ----------------------------------------------
    # Load dataset (With smart new file detection)
    # ----------------------------------------------
    
    # Extract the current file name from the uploader if it exists
    current_file_name = uploaded_file.name if hasattr(uploaded_file, "name") else "__from_session__"

    # Initialize the file name variable in session state if it does not exist
    if "loaded_file_name" not in st.session_state:
        st.session_state.loaded_file_name = None

    # CHANGE DETECTION: If no DataFrame is in memory OR the user uploaded a file with a different name
    if (st.session_state.get("df_shared") is None) or (current_file_name != st.session_state.loaded_file_name and current_file_name != "__from_session__"):
        
        # Load the brand-new CSV from scratch
        df = load_csv(uploaded_file)
        st.session_state.df_shared = df
        st.session_state.loaded_file_name = current_file_name  # Store the identity of the new file
        
        # FULL RESET: Safely clear states and generated reports from the previous dataset
        st.session_state.allow_ai = False
        st.session_state.target_selection = None
        st.session_state.target_selection_persistent = None  
        st.session_state.target_changed_alert = False 
        
        if 'target_shared' in st.session_state:
            del st.session_state['target_shared']
            
        if "ai_report_output" in st.session_state:
            del st.session_state["ai_report_output"]
            
        # Force an immediate rerun to compel visual components to redraw empty
        st.rerun()
        
    else:
        # If the user just returned from the AI page or changed the target, reuse the cached DataFrame
        df = st.session_state.df_shared

    # Guaranteed synchronization of the active state with the callback variable
    if st.session_state.get("target_selection_persistent") is not None:
        st.session_state.target_selection = st.session_state.target_selection_persistent

    # Initial onboarding messages (Only visible if there is genuinely no selection in memory)
    if st.session_state.get("target_selection_persistent") is None:
        col_msg1, col_msg2 = st.columns([1, 2.2])
        with col_msg1:
            st.success("Dataset uploaded successfully!", icon="✅")
        with col_msg2:
            st.info("Please select a target variable or 'None' to generate the EDA and enable the AI analysis.", icon="💡")


    # ----------------------------------------------
    # Target selector 
    # ----------------------------------------------
    with st.container(border=True):
        st.markdown("#### 🎯 Target variable")
        
        # Combine a 'None' option with the dataset column names
        all_options = ["None"] + list(df.columns)
        
        # Calculate the index if a persistent previous selection already exists in memory
        current_index = None
        if st.session_state.get("target_selection_persistent") in all_options:
            current_index = all_options.index(st.session_state.target_selection_persistent)

        # Synchronous callback function executed immediately in the backend upon widget state change
        def update_target_state():
            # Retrieve the newly selected value directly from the UI widget session key
            new_val = st.session_state.target_selector_ui
            
            if new_val is not None and new_val != st.session_state.target_selection_persistent:
                # If a previous valid selection existed, trigger the out-of-sync alert flag
                if st.session_state.target_selection_persistent is not None:
                    st.session_state.target_changed_alert = True
                
                # Update both runtime and cross-page persistent state variables synchronously
                st.session_state.target_selection = new_val
                st.session_state.target_selection_persistent = new_val

        # Render the selectbox bound to the isolated UI key and the background state callback
        st.selectbox(
            "Select a target variable (optional):",
            options=all_options,
            index=current_index,  
            placeholder="Choose a column or 'None'...",
            key="target_selector_ui",
            on_change=update_target_state
        )

        # Display the warning message if the target was modified after a report generation
        if st.session_state.get("target_changed_alert", False):
            st.warning(
                "⚠️ **Cambio de Target detectado:** El análisis actual en la página de la IA corresponde "
                "al target anterior. Por favor, presiona el botón **'✨ Analyze with AI'** de arriba si deseas "
                "generar un nuevo reporte adaptado a esta variable.",
                icon="🔄"
            )

    # Map the dropdown string selection to a strict Python object configuration (None or column string)
    selected_target = None if st.session_state.target_selection == "None" else st.session_state.target_selection


    # ----------------------------------------------
    # AI Page Redirection Logic    
    # ----------------------------------------------
    if ai_clicked and st.session_state.target_selection_persistent is not None:
        # Enable the authorization flag to grant access to the AI subpage via route validation
        st.session_state.allow_ai = True
        
        # Persist runtime variables globally so they can be consumed by the report backend
        st.session_state.target_shared = selected_target
        st.session_state.api_key_shared = st.session_state.user_api_key
        
        # Dismiss the yellow out-of-sync warning banner before transitioning pages
        st.session_state.target_changed_alert = False 
        
        # Trigger the native multipage redirection to the clean full-screen analysis layout
        st.switch_page("pages/ai_analysis.py")


    # ----------------------------------------------
    # Dataset preview
    # ----------------------------------------------        

    with st.container(border=True): 
        st.markdown("#### 🔍 Dataset Preview")
        st.dataframe(
            df.head(),
            use_container_width=True
        )
        
    # ----------------------------------------------
    # Metrics cards
    # ----------------------------------------------
    
    with st.container(border=True): 
        st.markdown("#### 🔢 Dataset Metrics")
        
        # Calculate full DataFrame memory footprint in Megabytes (MB)
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        
        # Count total number of perfectly identical duplicate records
        num_dups = df.duplicated().sum()
        
        # Split the metrics horizontally into a 4-column balanced grid layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container(border=True):
                # Format total row count with commas for high readability
                st.metric(label="Rows", value=f"{df.shape[0]:,}")
                
        with col2:
            with st.container(border=True):
                # Render the exact column features count
                st.metric(label="Columns", value=df.shape[1])
                
        with col3:
            with st.container(border=True):
                # Trigger a dynamic alert banner (inverse delta = red) if duplicate data threatens leakage
                st.metric(
                    label="Duplicates", 
                    value=num_dups,
                    delta=f"{num_dups} repeated" if num_dups > 0 else None,
                    delta_color="inverse"
                )
                
        with col4:
            with st.container(border=True):
                # Display total dataset memory footprint limited to two decimal spaces
                st.metric(label="Memory Usage", value=f"{memory_usage_mb:.2f} MB")



    # ----------------------------------------------
    # EDA Summary & Initial Analysis Block
    # ----------------------------------------------

    # Only render downstream exploratory data analysis once a target context has been set
    if st.session_state.get("target_selection_persistent") is not None:

        # --- Section Header: EDA Banner ---
        with st.container(border=True):
            # Centered layout using native HTML to cleanly separate the block from preceding widgets
            st.markdown(
                """
                <div style="text-align: center; padding: 10px 0;">
                    <h2 style="margin: 0; font-size: 30px; color: #1E293B;">📊 Exploratory Data Analysis</h2>
                    <p style="margin: 5px 0 0 0; font-size: 14px; color: #64748B;">
                        Explore distributions, correlations, and the structural integrity of your features.
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Execute statistical feature evaluation based on current file and selected label
        profile = profile_dataset(
            df=df,
            target=selected_target
        )

        # Convert raw statistical dictionary into a fluid, human-readable markdown narrative
        summary = generate_data_summary(
            profile
        )

        # --- Data Overview Narrative Panel ---
        with st.container(border=True):
            st.markdown("#### 🗒️ Data Overview")

            # Nested container structure provides an elegant premium card-like feel for text-heavy insights
            with st.container(border=True):
                st.markdown(summary)

        # --- Variables Summary Table ---
        with st.container(border=True):
            st.markdown("#### 📋 Variables Summary Table")
            
            # Fetch a tabular breakdown (data types, missing counts, etc.) of all available columns
            summary_table = generate_variable_summary_table(df)
            
            # Render the data matrix maximizing width and hiding the redundant numeric index track
            st.dataframe(summary_table, use_container_width=True, hide_index=True)


        
        # ----------------------------------------------
        # Visualizations (Organized in Tabs)
        # ----------------------------------------------

        # Define base layout views for cross-variable graphing
        tab_titles = ["📊 Distributions", "🔍 Quality & Structure", "🔗 Data Relations"]
        
        # Dynamically append a target analysis panel only when a target label is selected
        if selected_target:
            tab_titles.append("🎯 Target Analysis")

        # Instantiate native component tab array to reduce heavy vertical scrolling
        tabs = st.tabs(tab_titles)

        # --- TAB 1: DATA DISTRIBUTIONS ---
        with tabs[0]:
            # Numerical data frequency tracking via collapsible UI cards
            with st.expander("🔢 Numerical Variables Distribution", expanded=True):
                with st.container(border=True):
                    fig_num_dist = plot_numerical_distributions(df)
                    if fig_num_dist:
                        st.plotly_chart(fig_num_dist, use_container_width=True)
                    else:
                        st.info("No numerical columns available to plot distributions.")

            # Categorical string data frequency distribution tracking
            with st.expander("🔠 Categorical Variables Distribution "
                             "(only variables with ≤20 categories and >1 unique value)", 
                             expanded=True):
                with st.container(border=True):
                    fig_cat_dist = plot_categorical_distributions(df)
                    if fig_cat_dist:
                        st.plotly_chart(fig_cat_dist, use_container_width=True)
                    else:
                        st.info(
                            "No categorical columns available to plot distributions. "  
                            "Either none exist in the dataset or none meet the filter criteria "
                            "(≤20 categories and >1 unique value)."
                            )

        # --- TAB 2: QUALITY AND STRUCTURE ---
        with tabs[1]:
            # Render a dedicated evaluation panel for missing cell distributions
            with st.container(border=True):
                st.markdown("🔍 Missing Values Analysis")
                fig_missing = plot_missing_values(df)
                if fig_missing:
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.info("No missing values detected in this dataset.")

            # Evaluate string cardinality to flag keys, IDs, or high-dimensional text columns
            with st.container(border=True):
                st.markdown("🔠 Categorical Cardinality")
                fig_cardinality = plot_categorical_cardinality(df)
                if fig_cardinality:
                    st.plotly_chart(fig_cardinality, use_container_width=True)
                else:
                    st.info("No categorical columns detected.")
        
        # --- TAB 3: DATA RELATIONS ---
        with tabs[2]:
            # Linear association heatmap breakdown for baseline correlation mapping
            with st.container(border=True):
                st.markdown("####  Correlation Analysis")
                fig_corr = plot_correlation_heatmap(df)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough numerical columns.")

        # --- TAB 4: TARGET ANALYSIS (Conditional Panel) ---
        if selected_target:
            with tabs[3]:
                # Inspect how rows group, spread, or separate based on target boundaries
                with st.container(border=True):

                    st.markdown(f"🎯 Target Distribution (`{selected_target}`)")
                    fig_target = plot_target_distribution(df, selected_target)

                    if fig_target:
                        st.plotly_chart(fig_target, use_container_width=True)

                # Render advanced statistical evaluation (ANOVA/Pearson/Chi2) against the label
                with st.container(border=True):

                    st.markdown(f"#### 🎯 Predictive Feature Power against `{selected_target}`")
                    fig_target_corr = plot_target_correlations(df, selected_target)

                    if fig_target_corr:
                        st.plotly_chart(fig_target_corr, use_container_width=True)
                    else:
                        st.info("Could not calculate independent feature relationships for this target.")

                # Render advanced statistical evaluation (ANOVA/Pearson/Chi2) against the label
                with st.container(border=True):

                    st.markdown(f"#### 🎯 Predictive Feature Power against `{selected_target}`")
                    fig_target_corr = plot_target_correlations_2(df, selected_target)

                    if fig_target_corr:
                        st.plotly_chart(fig_target_corr, use_container_width=True)
                    else:
                        st.info("Could not calculate independent feature relationships for this target.")

