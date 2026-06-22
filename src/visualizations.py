"""
Visualization utilities.
"""
import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import f_oneway, chi2_contingency

# --------------------------------------------------
# Plot Numerical Distributions (Boxplot Grid)
# --------------------------------------------------

def plot_numerical_distributions(df: pd.DataFrame) -> go.Figure | None:
    """
    Create a dynamic grid of horizontal Boxplots for ALL numerical variables.
    Outliers are isolated as points to keep the visualization fully readable.
    """

    # Select only numerical columns
    cols_to_plot = df.select_dtypes(include="number").columns.tolist()
    
    # If no numerical columns exist, return None
    if not cols_to_plot:
        return None
        
    num_plots = len(cols_to_plot)
    
    # Use 2 columns if more than 1 variable, otherwise 1
    cols = 2 if num_plots >= 2 else 1
    rows = math.ceil(num_plots / 2)
    
    # Create subplot grid with titles for each variable
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=[f"Distribution of {c}" for c in cols_to_plot],
        vertical_spacing=0.6 / rows if rows > 1 else 0.4
    )
    
    # Iterate through each numerical column
    for i, col in enumerate(cols_to_plot):
        r = (i // 2) + 1
        c = (i % 2) + 1
        
        # Use horizontal boxplot for cleaner visualization
        box_fig = px.box(
            df, 
            x=col, 
            points="outliers",  # Show outliers as points, keep center clean
            color_discrete_sequence=["#0b61a4"]
        )
        
        # Add traces from Plotly Express figure into subplot grid
        for trace in box_fig.data:
            trace.showlegend = False 
            fig.add_trace(trace, row=r, col=c)
            
    # Calculate dynamic height (boxplots need less vertical space than histograms)
    calculated_height = max(250, rows * 180)
    
    # Update layout for readability
    fig.update_layout(
        title="Numerical Variables Distributions (Boxplot Analysis)",
        height=calculated_height,
        template="plotly_white",
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    # Remove redundant axis titles
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    
    return fig


# --------------------------------------------------
# Plot Categorical Distributions (Bar Chart Grid)
# --------------------------------------------------

def plot_categorical_distributions(df: pd.DataFrame) -> go.Figure | None:
    """
    Create a grid of bar charts for categorical variables with reasonable cardinality.
    High-cardinality columns (IDs, unique names) are excluded to keep plots readable.
    """

    # 1. Select categorical variables (object, category, boolean types)
    categorical_cols = df.select_dtypes(include=["object", "category", "boolean"]).columns.tolist()
    
    # Smart filter: exclude columns with too many categories (>20) or only one unique value
    valid_cols = [c for c in categorical_cols if df[c].nunique() <= 20 and df[c].nunique() > 1]
    
    # If no valid categorical columns exist, return None
    if not valid_cols:
        return None
        
    # Limit to a maximum of 4 categorical variables for clarity
    cols_to_plot = valid_cols[:4]
    num_plots = len(cols_to_plot)
    
    # Configure subplot grid (rows/cols based on number of variables)
    rows = 2 if num_plots > 2 else 1
    cols = 2 if num_plots >= 2 else 1
    
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=[f"Value Counts of {c}" for c in cols_to_plot]
    )
    
    # Compute frequencies and fill subplots
    for i, col in enumerate(cols_to_plot):
        r = (i // 2) + 1
        c = (i % 2) + 1
        
        # Get frequency counts sorted by value
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, 'count']
        
        # Create vertical bar chart for each categorical variable
        bar_fig = px.bar(counts, x=col, y='count', color=col, color_discrete_sequence=["#0b61a4"])
        
        # Add traces from Plotly Express figure into subplot grid
        for trace in bar_fig.data:
            trace.showlegend = False
            fig.add_trace(trace, row=r, col=c)
            
    # Global layout adjustments for readability
    fig.update_layout(
        title="Categorical Variables Distributions",
        height=350 if rows == 1 else 600,
        template="plotly_white",
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    return fig




# --------------------------------------------------
# Plot Missing Values (Bar Chart)
# --------------------------------------------------

def plot_missing_values(df: pd.DataFrame) -> go.Figure | None:
    """
    Create a bar chart showing the percentage of missing values per column.
    """

    # Calculate percentage of missing values per column
    missing = (
        df.isna()
        .mean()
        .mul(100)              # Convert to percentage
        .sort_values(ascending=False)
    )

    # Keep only columns with >0% missing values
    missing = missing[missing > 0]

    # If no missing values exist, return None
    if missing.empty:
        return None  

    # Create bar chart with Plotly Express
    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "Column", "y": "Missing (%)"},
        color_discrete_sequence=["#0b61a4"],
        title="Missing Values by Column"
    )

    # Return figure for visualization
    return fig



# --------------------------------------------------
# Plot Correlation Heatmap (Numerical Variables)
# --------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """
    Create a correlation heatmap for numerical columns.
    """

    # Select only numerical columns
    numerical_df = df.select_dtypes(include="number")

    # If fewer than 2 numerical columns exist, correlation cannot be computed
    if numerical_df.shape[1] < 2:
        return None

    # Compute correlation matrix (default = Pearson correlation)
    corr_matrix = numerical_df.corr(numeric_only=True)

    # Create heatmap with correlation values annotated
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",   # Show correlation values with 2 decimal precision
        aspect="auto",     # Adjust aspect ratio automatically
        title="Correlation Heatmap",
        color_continuous_scale="RdBu",   
        zmin=-1, zmax=1,                 
        template="plotly_white"
    )

    fig.update_layout(
        margin=dict(t=60, b=40, l=40, r=40),
        coloraxis_colorbar=dict(title="Correlation")
    )

    # Return figure for visualization
    return fig



# --------------------------------------------------
# Plot Target Distribution (Numeric & Categorical)
# --------------------------------------------------

def plot_target_distribution(df: pd.DataFrame, target: str) -> go.Figure | None:
    """
    Create a distribution chart for the target variable.
    Handles both numerical and categorical targets.
    """

    # Validate target input
    if target is None:
        return None
    if target not in df.columns:
        return None

    series = df[target]

    # Numeric target → histogram
    if pd.api.types.is_numeric_dtype(series):
        fig = px.histogram(
            df,
            x=target,
            nbins=30,  # Default bin count for continuous distribution
            title=f"{target} Distribution",
            color_discrete_sequence=["#0b61a4"]
        )
        return fig

    # Categorical target → bar chart of frequencies
    counts = (
        series
        .value_counts(dropna=False)  # Include NaN categories if present
        .reset_index()
    )
    counts.columns = [target, "count"]

    fig = px.bar(
        counts,
        x=target,
        y="count",
        title=f"{target} Distribution",
        color_discrete_sequence=["#0b61a4"]
    )

    # Return figure for visualization
    return fig


# --------------------------------------------------
# Plot Categorical Cardinality (Unique Values per Variable)
# --------------------------------------------------

def plot_categorical_cardinality(df: pd.DataFrame) -> go.Figure | None:
    """
    Create a bar chart showing the cardinality (number of unique values)
    for each categorical variable in the dataset.
    """

    # Select only non-numeric columns (categorical/boolean/string types)
    categorical_df = df.select_dtypes(exclude="number")

    # If no categorical variables exist, return None
    if categorical_df.empty:
        return None
    
    # Compute cardinality (unique values per column), sorted descending
    cardinality = categorical_df.nunique().sort_values(ascending=False)

    # Create bar chart with Plotly Express
    fig = px.bar(
        x=cardinality.index,
        y=cardinality.values,
        labels={"x": "Categorical Variable", "y": "Unique Values"},
        title="Categorical Variables Cardinality",
        color_discrete_sequence=["#0b61a4"]
    )

    # eturn figure for visualization
    return fig


# --------------------------------------------------
# Generate Variable Summary Table (Numeric & Categorical)
# --------------------------------------------------

def generate_variable_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table for all variables (numeric + categorical).
    Includes type, missing percentage, unique values, most frequent value,
    frequency of mode, and a simple classification (Numerical vs Categorical).
    """

    # Base summary: type, missing %, unique values
    summary = pd.DataFrame({
        "Variable": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing (%)": df.isna().mean().mul(100).round(2),
        "Unique Values": df.nunique()
    })

    # Mode and frequency of mode
    modes = []
    mode_freqs = []
    for col in df.columns:
        if df[col].dropna().empty:
            # If column is entirely empty, append None
            modes.append(None)
            mode_freqs.append(None)
        else:
            # Most frequent value (mode)
            mode_val = df[col].mode().iloc[0]
            modes.append(mode_val)
            # Frequency of mode as percentage
            freq = df[col].value_counts(normalize=True).max() * 100
            mode_freqs.append(round(freq, 2))

    summary["Most Frequent Value"] = modes
    summary["Frequency of Mode (%)"] = mode_freqs

    # Simple classification: Numerical vs Categorical
    summary["Data Type Category"] = [
        "Numerical" if pd.api.types.is_numeric_dtype(df[col]) else "Categorical"
        for col in df.columns
    ]

    # Reset index to avoid duplicate column names
    summary.reset_index(drop=True, inplace=True)

    # Return summary DataFrame
    return summary


# --------------------------------------------------
# Plot Target Correlations (Predictive Association Analysis)
# --------------------------------------------------

def plot_target_correlations(df: pd.DataFrame, target: str) -> go.Figure | None:
    """
    Calculate the predictive association strength between features and the target.
    Automatically detects if the target should be treated as continuous (regression)
    or discrete classes (classification, e.g., Titanic Survived).
    """

    # Validate target input
    if not target or target not in df.columns:
        return None

    features_importance = []
    
    # Detect target type
    # If numeric but with ≤10 unique values (e.g., 0/1), treat as categorical
    is_target_numeric = pd.api.types.is_numeric_dtype(df[target])
    if is_target_numeric and df[target].nunique() <= 10:
        is_target_numeric = False  # Force classification flow

    # --- SCENARIO A: CLASSIFICATION CONTEXT ---
    if not is_target_numeric:
        target_classes = df[target].dropna().unique()
        if len(target_classes) < 2:
            return None

        for col in df.columns:
            if col == target:
                continue
                
            # Numerical feature vs categorical target → ANOVA
            if pd.api.types.is_numeric_dtype(df[col]):
                groups = [df[df[target] == c][col].dropna() for c in target_classes]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) > 1:
                    f_stat, p_val = f_oneway(*groups)
                    strength = 1.0 - p_val if not np.isnan(p_val) else 0.0
                    features_importance.append({
                        "Feature": col,
                        "Association Power": min(1.0, max(0.0, strength)),
                        "Type": "Numerical (ANOVA)"
                    })

            # Categorical feature vs categorical target → Chi-Square
            else:
                contingency_table = pd.crosstab(df[col], df[target])
                if contingency_table.size > 1:
                    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                    strength = 1.0 - p_val
                    features_importance.append({
                        "Feature": col,
                        "Association Power": min(1.0, max(0.0, strength)),
                        "Type": "Categorical (Chi2)"
                    })

    # --- SCENARIO B: REGRESSION CONTEXT ---
    else:
        numerical_df = df.select_dtypes(include="number")
        if numerical_df.shape[1] >= 2:
            correlations = numerical_df.corr(numeric_only=True)[target].drop(target).abs()
            for col, val in correlations.items():
                features_importance.append({
                    "Feature": col,
                    "Association Power": val,
                    "Type": "Numerical (Pearson)"
                })

    # Build results DataFrame
    res_df = pd.DataFrame(features_importance)
    if res_df.empty:
        return None
        
    res_df = res_df.sort_values(by="Association Power", ascending=True)

    # Create interactive bar chart
    title_suffix = "Classification Context" if not is_target_numeric else "Regression Context"
    fig = px.bar(
        res_df,
        x="Association Power",
        y="Feature",
        color="Type",  # Color by statistical method
        orientation="h",
        title=f"Feature Predictive Association with Target: '{target}' ({title_suffix})",
        labels={"Association Power": "Statistical Association Score (Higher = Stronger Relationship)"},
        color_discrete_map={
            "Numerical (ANOVA)": "#0b61a4",
            "Categorical (Chi2)": "#2e7d32",
            "Numerical (Pearson)": "#475569"
        }
    )

    # Layout adjustments
    fig.update_layout(
        template="plotly_white",
        height=max(300, len(res_df) * 45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# --------------------------------------------------
# Plot Target Correlations (Predictive Association Analysis)
# --------------------------------------------------

def plot_target_correlations_2(df: pd.DataFrame, target: str) -> go.Figure | None:
    """
    Calculate the predictive association strength between features and the target.
    Automatically detects if the target should be treated as continuous (regression)
    or discrete classes (classification, e.g., Titanic Survived).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str
        Target column name.

    Returns
    -------
    go.Figure | None
        Plotly figure showing association statistics, or None if not applicable.
    """

    # Validate target input
    if not target or target not in df.columns:
        return None

    features_importance = []
    
    # Detect target type
    is_target_numeric = pd.api.types.is_numeric_dtype(df[target])
    if is_target_numeric and df[target].nunique() <= 10:
        is_target_numeric = False  # Force classification flow

    # --- SCENARIO A: CLASSIFICATION CONTEXT ---
    if not is_target_numeric:
        target_classes = df[target].dropna().unique()
        if len(target_classes) < 2:
            return None

        for col in df.columns:
            if col == target:
                continue
                
            # Numerical feature vs categorical target → ANOVA
            if pd.api.types.is_numeric_dtype(df[col]):
                groups = [df[df[target] == c][col].dropna() for c in target_classes]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) > 1:
                    f_stat, p_val = f_oneway(*groups)
                    features_importance.append({
                        "Feature": col,
                        "Statistic": f_stat,
                        "p_value": p_val,
                        "Type": "Numerical (ANOVA)"
                    })

            # Categorical feature vs categorical target → Chi-Square
            else:
                contingency_table = pd.crosstab(df[col], df[target])
                if contingency_table.size > 1:
                    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                    features_importance.append({
                        "Feature": col,
                        "Statistic": chi2,
                        "p_value": p_val,
                        "Type": "Categorical (Chi2)"
                    })

    # --- SCENARIO B: REGRESSION CONTEXT ---
    else:
        numerical_df = df.select_dtypes(include="number")
        if numerical_df.shape[1] >= 2:
            # Pearson correlation
            pearson_corr = numerical_df.corr(numeric_only=True)[target].drop(target)
            for col, val in pearson_corr.items():
                features_importance.append({
                    "Feature": col,
                    "Statistic": val,
                    "p_value": None,  # Pearson correlation does not directly provide p-value here
                    "Type": "Numerical (Pearson)"
                })
            # Spearman correlation
            spearman_corr = numerical_df.corr(method="spearman")[target].drop(target)
            for col, val in spearman_corr.items():
                features_importance.append({
                    "Feature": col,
                    "Statistic": val,
                    "p_value": None,  # Spearman correlation does not directly provide p-value here
                    "Type": "Numerical (Spearman)"
                })

    # Build results DataFrame
    res_df = pd.DataFrame(features_importance)
    if res_df.empty:
        return None
        
    # Sort by absolute statistic for ranking
    res_df = res_df.sort_values(by="Statistic", ascending=True)

    # Create interactive bar chart
    title_suffix = "Classification Context" if not is_target_numeric else "Regression Context"
    fig = px.bar(
        res_df,
        x="Statistic",
        y="Feature",
        color="Type",
        orientation="h",
        title=f"Feature Association with Target: '{target}' ({title_suffix})",
        labels={"Statistic": "Association Statistic (magnitude)"},
        color_discrete_map={
            "Numerical (ANOVA)": "#0b61a4",
            "Categorical (Chi2)": "#2e7d32",
            "Numerical (Pearson)": "#475569",
            "Numerical (Spearman)": "#9c27b0"
        },
        hover_data=["p_value"]  # Show p-value in tooltip when available
    )

    # Layout adjustments
    fig.update_layout(
        template="plotly_white",
        height=max(300, len(res_df) * 45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig





