# Machine Learning Functions
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
# ML PARAMTERS 
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

# ==============================================================================
# MODEL EVALUATION
# ==============================================================================
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

def save_shap_summary(model, X, title, filepath):
    """
    Generate and save a SHAP summary plot.

    Parameters
    ----------
    model : fitted model
        Trained XGBoost model.

    X : pandas.DataFrame
        Feature matrix used to calculate SHAP values.

    title : str
        Plot title.

    filepath : str
        Output file path.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
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