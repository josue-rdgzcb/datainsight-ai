"""
Dataset profiling utilities.

"""

from typing import Any
import pandas as pd
import numpy as np



# --------------------------------------------------
# Dataset Profiling & Metadata Extraction Engines
# --------------------------------------------------
def profile_dataset(
    df: pd.DataFrame,
    target: str | None = None
) -> dict[str, Any]:
    """
    Generate an advanced, feature-rich profile of a dataset for LLM ingestion.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    target : str | None, default=None
        Selected target column.

    Returns
    -------
    dict[str, Any]
        Comprehensive statistical and structural metadata payload.
    """
    # Calculate the absolute global volume of cells in the matrix
    total_cells = df.shape[0] * df.shape[1]

    # Count the total number of empty or null coordinates across all axes
    missing_cells = int(df.isna().sum().sum())

    # Establish relative volume tracking for data gaps
    missing_percentage = (
        (missing_cells / total_cells) * 100
        if total_cells > 0
        else 0
    )

    # Compute a standard completeness score matrix ratio (0.0 to 1.0)
    completeness = (
        1 - (missing_cells / total_cells)
        if total_cells > 0
        else 1
    )

    # Segment specific tracking domains based on strict numeric boundaries
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Attempt to parse and isolate potential datetime tracks before text matching
    datetime_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # Extract object, boolean, and category types while excluding numeric/datetime signatures
    categorical_columns = [
        c for c in df.columns 
        if c not in numerical_columns and c not in datetime_columns
    ]

    # Calculate full RAM usage profile accounting for deep object strings
    memory_usage_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    # Compute individual feature missing ratios to flag bad quality tracks
    missing_by_column = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    # Isolate the top 3 highest-impact broken features for prompt focus
    top_missing_columns = (
        missing_by_column[missing_by_column > 0]
        .head(3)
        .round(2)
        .to_dict()
    )

    # --- NEW ADVANCED METRICS FOR ENRICHED AI CONTEXT ---
    
    # Constant Columns Detection: Find features with zero variance (useless for ML algorithms)
    constant_columns = [c for c in df.columns if df[c].nunique(dropna=True) == 1]

    # High Cardinality Flags: Categorical features with over 20 unique values (overfitting risk)
    high_cardinality_features = [
        c for c in categorical_columns 
        if df[c].nunique(dropna=True) > 20 and c != target
    ]

    # Interquartile Range (IQR) Outliers Detector (Evaluates numerical noise profiles)
    total_outlier_rows = 0
    if numerical_columns:
        outlier_mask = pd.Series(False, index=df.index)
        for col in numerical_columns:
            if col == target:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                # Flag values that sit outside 1.5 times the IQR distance boundaries
                outlier_mask |= (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
        total_outlier_rows = int(outlier_mask.sum())

    outliers_percentage = (total_outlier_rows / df.shape[0] * 100) if df.shape[0] > 0 else 0

    # High Correlation Detection (Identifica multicolinealidad para la IA)
    highly_correlated_pairs = []
    if len(numerical_columns) >= 2:
        # Calculamos la matriz absoluta para capturar tanto relaciones muy positivas como muy negativas
        corr_matrix = df[numerical_columns].corr(numeric_only=True).abs()
        
        # Recorremos solo la mitad superior de la matriz para evitar duplicar pares (ej: var1-var2 y var2-var1)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col in upper_tri.columns:
            # Buscamos coeficientes mayores o iguales a 0.70 (estándar de colinealidad fuerte)
            high_corr_features = upper_tri[upper_tri[col] >= 0.70][col].index.tolist()
            for feature in high_corr_features:
                if col != target and feature != target: # Ignoramos el target aquí, nos interesan las independientes
                    coef = df[col].corr(df[feature])
                    highly_correlated_pairs.append(f"`{feature}` vs `{col}` (r = {coef:.2f})")


    # Assemble the enriched statistical metadata dictionary payload
    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": round(memory_usage_mb, 2),
        "numerical_columns_count": len(numerical_columns),
        "categorical_columns_count": len(categorical_columns),
        "datetime_columns_count": len(datetime_columns),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "constant_columns": constant_columns,
        "high_cardinality_categorical_columns": high_cardinality_features,
        "missing_cells": missing_cells,
        "missing_percentage": round(missing_percentage, 2),
        "completeness": round(completeness, 3),
        "top_missing_columns": top_missing_columns,
        "outlier_rows_count": total_outlier_rows,
        "outliers_percentage": round(outliers_percentage, 2),
        "highly_correlated_features": highly_correlated_pairs, 
        # Evaluate target metadata contextually if the user selected a valid label
        "target_info": analyze_target(df, target) if target else None,
    }

    return profile



# --------------------------------------------------
# Target Feature Context & Distribution Analyzers
# --------------------------------------------------

def analyze_target(
    df: pd.DataFrame,
    target: str
) -> dict[str, Any]:
    """
    Analyze the selected target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    target : str
        Target column name.

    Returns
    -------
    dict[str, Any]
        Target analysis results.
    """
    # Isolate the targeted column track as a separate Pandas Series
    series = df[target]

    # Calculate individual target column missing cell ratio
    missing_percentage = series.isna().mean() * 100

    # Determine whether to use a numerical regression or categorical classification layout
    is_numeric = pd.api.types.is_numeric_dtype(series)

    # --- ESCENARIO A: CONTINUOUS TARGET VARIABLES (Regression Engine Context) ---
    if is_numeric:
        # Validate that the sequence contains data points before computing metrics to avoid NaN crashes
        has_data = not series.dropna().empty
        return {
            "name": target,
            "dtype": "numerical",
            "missing_percentage": round(missing_percentage, 2),
            "unique_values": int(series.nunique()),
            "mean": round(float(series.mean()), 2) if has_data else None,
            "median": round(float(series.median()), 2) if has_data else None,
            "skewness": round(float(series.skew()), 2) if has_data else None,
            "min": round(float(series.min()), 2) if has_data else None,
            "max": round(float(series.max()), 2) if has_data else None,
        }

    # --- ESCENARIO B: CATEGORICAL TARGET VARIABLES (Classification Engine Context) ---
    # Extract structural balance metrics to identify representation skew
    value_distribution = series.value_counts(
        normalize=True,
        dropna=True
    )

    # Capture the maximum class ratio to evaluate severe imbalance states
    max_class_ratio = (
        float(value_distribution.max())
        if not value_distribution.empty
        else 0
    )

    # Rule of Thumb: Consider a classification target imbalanced if a single class owns >= 70% of rows
    balanced = max_class_ratio < 0.70

    return {
        "name": target,
        "dtype": "categorical",
        "missing_percentage": round(missing_percentage, 2),
        "unique_values": int(series.nunique()),
        "balanced": balanced,
        "largest_class_ratio": round(max_class_ratio, 2),
        # Isolate the index point of the absolute mode to extract a string object value
        "mode": series.mode().iloc[0] if not series.dropna().empty else None,
    }


# --------------------------------------------------
# Automated EDA Narrative & Insights Generation
# --------------------------------------------------
def generate_data_summary(
    profile: dict[str, Any]
) -> str:
    """
    Generate an automatic EDA summary with natural and fluid narrative.

    Parameters
    ----------
    profile : dict[str, Any]
        A dictionary containing dataset profile metadata and statistics.

    Returns
    -------
    str
        A human-readable markdown summary block.
    """
    summary = []

    # 1. Structural dimensions and overall feature space segmentation
    summary.append(
        f"**Dataset Overview:** The dataset comprises **{profile['rows']:,} rows** and **{profile['columns']} columns**, "
        f"structuring a feature space of **{profile['numerical_columns_count']} numerical** and "
        f"**{profile['categorical_columns_count']} categorical** variables."
    )

    # 2. Data Quality evaluation (Fusing missing values and duplicate checks into one narrative)
    # Normalize completeness score calculation dynamically to ensure a 0-100% boundary framework
    completeness_pct = profile['completeness'] * 100 if profile['completeness'] <= 1.0 else profile['completeness']
    
    missing_text = (
        f"Data integrity is exceptionally high with **{completeness_pct:.1f}% completeness** (only {profile['missing_percentage']:.2f}% missing cells)."
        if profile['completeness'] >= 0.95 else
        f"The dataset shows a **{completeness_pct:.1f}% completeness rate**, meaning **{profile['missing_percentage']:.2f}% of the total cells are missing**."
    )
    
    # Isolate missingness features if a clear high-ratio track is present
    if profile["top_missing_columns"]:
        missing_cols = ", ".join([f"`{col}`" for col in profile["top_missing_columns"].keys()])
        missing_text += f" Notably, missingness is concentrated in: {missing_cols}."

    # Process identical row records to flag potential evaluation leaks
    dup_count = profile['duplicates']
    dup_text = (
        "Fortunately, **no duplicate records** were detected."
        if dup_count == 0 else
        f"Additionally, **{dup_count:,} duplicate rows** were found, which should be handled to prevent data leakage."
    )
    
    summary.append(f"**Data Quality & Integrity:** {missing_text} {dup_text}")

    # 3. Dedicated Target variable validation (Classification vs Regression layout contexts)
    target_info = profile.get("target_info")
    if target_info:
        target_name = f"`{target_info['name']}`"
        missing_tgt = f" (with {target_info['missing_percentage']:.1f}% missing values)" if target_info['missing_percentage'] > 0 else ""
        
        # Scenario A: The target vector acts as a discrete set of label classes
        if target_info["dtype"] == "categorical":
            balance_text = "exhibits a **balanced distribution**" if target_info["balanced"] else "presents a **class imbalance**"
            summary.append(
                f"**Target Analysis:** The designated target {target_name}{missing_tgt} is **categorical** with **{target_info['unique_values']} distinct classes**. "
                f"The majority class is **'{target_info['mode']}'**, and the target overall {balance_text} across categories."
            )
        # Scenario B: The target vector acts as a continuous numeric distribution scale
        else:
            # Format float attributes explicitly to preserve mathematical dashboard legibility
            v_mean = f"{target_info['mean']:.2f}" if isinstance(target_info['mean'], float) else target_info['mean']
            v_med = f"{target_info['median']:.2f}" if isinstance(target_info['median'], float) else target_info['median']
            
            summary.append(
                f"**Target Analysis:** The designated target {target_name}{missing_tgt} is **continuous (numerical)** spanning "
                f"from **{target_info.get('min')}** to **{target_info.get('max')}** (Range). "
                f"It features a **mean of {v_mean}** and a **median of {v_med}** across its {target_info['unique_values']:,} unique values."
            )

    # 4. Final analytical verdict and modeling readiness tracking
    # Rule of Thumb: A dataset is suitable if completeness exceeds 80% and duplicate ratio is under 5%
    if profile["completeness"] > 0.8 and profile["duplicates"] < 0.05 * profile["rows"]:
        summary.append(
            "**Verdict:** The dataset is in solid condition and appears **fully suitable** for immediate exploratory data analysis (EDA) and machine learning workflows."
        )
    else:
        summary.append(
            "**Verdict:** Initial signs indicate the dataset **requires preprocessing** (such as deduplication, imputation, or scaling) before proceeding to modeling."
        )

    # Join the textual sequence tracks into a uniform markdown narrative structure
    return "\n\n".join(summary)
