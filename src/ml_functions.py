# Machine Learning Functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import shap

# ==============================================================================
# CLEAN DATA 
# ==============================================================================
def clean_column_names(df):
    """
    Standardize DataFrame column names by:
    - converting to string
    - replacing non-alphanumeric characters with underscores
    - collapsing consecutive underscores
    - removing leading/trailing underscores
    """
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return df

def filter_rare_groups(
    df,
    group_col="receiver_country",
    min_obs=50
):
    """
    Remove groups with fewer than min_obs observations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    group_col : str
        Column used to define groups.

    min_obs : int
        Minimum number of observations required.

    Returns
    -------
    pd.DataFrame
        Filtered dataset.
    """
    before = df[group_col].nunique()

    filtered_df = df.groupby(group_col).filter(
        lambda x: len(x) >= min_obs
    )

    after = filtered_df[group_col].nunique()

    print(
        f"Removed {before - after} {group_col} categories "
        f"with fewer than {min_obs} observations"
    )

    return filtered_df

# ==============================================================================
# ML 
# ==============================================================================
xgb_params = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=99
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=99
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def split_train_test(
    X,
    y,
    test_size=0.2,
    random_state=99,
    model_name="Model"
):
    """
    Split features and target into train and test sets
    and print dataset dimensions.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"{model_name} Feature Training Data Shape: {X_train.shape}")
    print(f"{model_name} Feature Test Data Shape: {X_test.shape}")
    print(f"{model_name} Target Training Data Shape: {y_train.shape}")
    print(f"{model_name} Target Test Data Shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

# ==============================================================================
# MODEL EVALUATION
# ==============================================================================
def evaluate_cv(
    model,
    X_train,
    y_train,
    cv,
    model_name="Model"
):
    """
    Evaluate a regression model using cross-validation
    and print average scores for MAE, RMSE, and R².
    """

    metrics = [
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
        "r2"
    ]

    results = {}

    for metric in metrics:

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=metric
        )

        mean_score = np.mean(scores)

        # Convert sklearn's negative error metrics back to positive
        if metric.startswith("neg_"):
            mean_score = -mean_score

        print(
            f"{model_name} Training Data: "
            f"{metric} = {mean_score:.3f}"
        )

        results[metric] = mean_score

    return results

def evaluate_regression_model(y_true, y_pred, model_name="Model"):
    """
    Calculate and print regression metrics.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Predicted values.

    model_name : str
        Name of the model for reporting.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, and R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} Test Data: MAE = {mae:.3f}")
    print(f"{model_name} Test Data: RMSE = {rmse:.3f}")
    print(f"{model_name} Test Data: R² = {r2:.3f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    }

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================
def save_actual_vs_predicted_plot(
    y_true,
    y_pred,
    model_name,
    filepath
):
    """
    Create and save an Actual vs Predicted scatter plot.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Predicted values.

    model_name : str
        Name of model to display in title.

    filepath : str
        Output file path.
    """

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red",
        linestyle="--"
    )

    plt.xlabel("Actual Impact Score")
    plt.ylabel("Predicted Impact Score")
    plt.title(
        f"{model_name}: Actual vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def save_shap_summary(
    model,
    X,
    title,
    filepath,
    rename_dict=None
):
    """
    Generate and save a SHAP summary plot.

    Parameters
    ----------
    model : fitted model
        Trained XGBoost model.

    X : pd.DataFrame
        Feature matrix used to calculate SHAP values.

    title : str
        Plot title.

    filepath : str
        Output file path.

    rename_dict : dict, optional
        Dictionary mapping original feature names
        to display names for the SHAP plot.
    """

    # SHAP must use the original feature matrix
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Create display copy for the plot
    X_display = X.copy()

    if rename_dict is not None:
        X_display = X_display.rename(
            columns=rename_dict
        )

    shap.summary_plot(
        shap_values,
        X_display,
        show=False
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()