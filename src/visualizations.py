"""
Visualization utilities.
"""
import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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
# Feature Importance Analysis
# --------------------------------------------------

def plot_feature_relevance(
    df: pd.DataFrame,
    target: str,
    top_n: int = 10
) -> go.Figure | None:
    """
    Generate feature relevance analysis based on the selected target.

    Classification:
    Uses Mutual Information Classification.

    Regression:
    Uses Mutual Information Regression.

    Handles:
    - Numerical features
    - Categorical features
    - Missing values

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    target : str
        Target variable.

    top_n : int
        Number of top features displayed.

    Returns
    -------
    go.Figure | None
        Plotly feature importance chart.
    """

    # Validate target
    if target is None or target not in df.columns:
        return None


    # Copy dataframe
    data = df.copy()


    # Remove rows where target is missing
    data = data.dropna(subset=[target])

    if data.empty:

        return None


    X = data.drop(columns=[target])

    y = data[target]


    # --------------------------------------------------
    # Detect problem type
    # --------------------------------------------------

    is_classification = (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10)

    # --------------------------------------------------
    # Detect feature types
    # --------------------------------------------------

    categorical_features = (
        X.select_dtypes(exclude="number").columns.tolist()
    )

    numerical_features = (
        X.select_dtypes(include="number").columns.tolist()
    )


    if (len(categorical_features) == 0 and len(numerical_features) == 0):

        return None


    # --------------------------------------------------
    # Preprocessing pipeline
    # --------------------------------------------------

    transformers = []


    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )


    if numerical_features:

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        transformers.append(
            ("numerical", numerical_pipeline, numerical_features)
        )


    preprocessor = ColumnTransformer(transformers=transformers)

    X_encoded = preprocessor.fit_transform(X)

    feature_names = (preprocessor.get_feature_names_out())

    # --------------------------------------------------
    # Calculate importance
    # --------------------------------------------------

    if is_classification:

        importance = mutual_info_classif(
            X_encoded,
            y,
            random_state=42
        )

        method = "Mutual Information Classification"

    else:

        importance = mutual_info_regression(
            X_encoded,
            y,
            random_state=42
        )

        method = "Mutual Information Regression"


    title = (
        f"Feature Relevance Analysis"
        f"<br><sup>"
        f"Target: {target} | Method: {method}"
        f"</sup>"
    )

    # --------------------------------------------------
    # Build importance dataframe
    # --------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "Encoded Feature": feature_names,
            "Importance": importance
        }
    )


    def get_original_feature(encoded_name: str) -> str:
        """
        Recover the original feature name after preprocessing.
        """

        # Numerical features
        if encoded_name.startswith("numerical__"):
            return encoded_name.replace(
                "numerical__",
                ""
            )

        # Categorical features
        if encoded_name.startswith("categorical__"):

            feature = encoded_name.replace(
                "categorical__",
                ""
            )

            # Find which original categorical feature generated it
            for col in categorical_features:
                prefix = f"{col}_"

                if feature.startswith(prefix):
                    return col

            return feature

        return encoded_name


    importance_df["Feature"] = (
        importance_df["Encoded Feature"]
        .apply(get_original_feature)
    )


    importance_df = (importance_df.groupby("Feature", as_index=False)["Importance"].sum())


    importance_df = (importance_df[importance_df["Importance"] > 0].sort_values(by="Importance", ascending=False).head(top_n))


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=True
        )
    )


    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        labels={
            "Importance": "Mutual Information Score",
            "Feature": ""
        }
    )


    fig.update_layout(
        template="plotly_white",
        height=max(
            400,
            len(importance_df) * 40
        )
    )


    return fig