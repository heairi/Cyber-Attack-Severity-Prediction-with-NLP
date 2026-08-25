# ==============================================================================
# COMBINING NLP + ML TO PREDICT CYBER ATTACK SEVERITY
# ==============================================================================
# Predict the severity of a cyber incident (combined financial, political, 
# intelligence, and funtional impact) based on 3 XGRegressor models:
#
# Model A: pre-existing cyber operation categories (data theft, data theft & doxing,
# hijacing with misuse, hijacking without misuse, ransomware) + control features 
# (initiator country, receiver country, cyber_conflict_issue, offline_conflict_intensity,
# affected entities value, affected third countries value)
# 
# Model B: latent cyber operation categories extracted using NLP (BERTopic anlalysis:
# website defacement & hacktivism, malware & cyber espionage, ransomware & cyber extortion
# DDoS & service disruption campaigns, data breaches & information exposure, institutional
# ransomware attacks, vulnerability exploitation & theft, surveillance & mobile espionage, 
# government intelligence operations) + control features
# 
# Model C: combined (pre-exisiting cyber operation categories +  latent cyber operation categories
# + control features)

# ==============================================================================
# IMPORT PACKAGES
# ==============================================================================
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from src.ml_functions import (
    clean_column_names,
    filter_rare_groups,
    xgb_params,
    cv,
    evaluate_cv,
    evaluate_regression_model,
    split_train_test,
    save_actual_vs_predicted_plot,
    save_shap_summary)
import matplotlib.pyplot as plt
import shap
import pickle

# ==============================================================================
# LOAD DATA 
# ==============================================================================
# Dyadic data with BERTopic latent cyber operation categories 
PROJECT_DIR = Path(__file__).parent
dyadic_data_topic = pd.read_csv(
    PROJECT_DIR / "data" / "eurepoc_dataset" / "dyadic_data_topics.csv"
)

# Original dyadic data
url = "https://zenodo.org/records/14965395/files/eurepoc_dyadic_dataset_0_1.csv?download=1"

dyadic_data_orig = pd.read_csv(url)

# ==============================================================================
# CLEAN DATA 
# ==============================================================================
# The topic analysis removed topic outliers (assigned -1) and short word descriptions (n < 5)
# Filter dyadic_data_orig so they're the same in order to compare models 
common_ids = set(
    dyadic_data_topic["incident_id"]
)

dyadic_data_orig_matched = dyadic_data_orig[
    dyadic_data_orig["incident_id"].isin(common_ids)
].copy()

print(f"BERTopic Dyadic Dataset Shape:  {dyadic_data_topic.shape}")
print(f"Original Dyadic Dataset Shape: {dyadic_data_orig_matched.shape}")

# Drop countries that only appear a few times in the dataset 
# as they will be very difficult to predict
filtered_topic = filter_rare_groups(
    dyadic_data_topic,
    group_col="receiver_country",
    min_obs=50
)

filtered_orig = filter_rare_groups(
    dyadic_data_orig_matched,
    group_col="receiver_country",
    min_obs=50
)

# ==============================================================================
# MODEL A: EXISTING CYBER OPERATION CATEGORIES ONLY
# ==============================================================================
existing_features = [
    "Data theft",
    "Data theft & Doxing",
    "Hijacking with Misuse",
    "Hijacking without Misuse",
    "Ransomware"
]

control_features = [
    "initiator_country",
    "receiver_country",
    "cyber_conflict_issue",
    "offline_conflict_intensity",
    "affected_entities_value",
    "affected_third_countries_value"
]

 # One-hot encode categorical variables
X_controls = pd.get_dummies(
    filtered_orig[control_features],
    drop_first=True
)

X_existing = pd.concat(
    [
        X_controls,
        filtered_orig[existing_features]
    ],
    axis=1
)

# Fix columns with special characters
X_existing = clean_column_names(X_existing)

# encode y labels to integers
y = filtered_orig["impact_indicator_score"]

# Split into training and testing data
X_train_A, X_test_A, y_train_A, y_test_A = split_train_test(
    X_existing,
    y,
    model_name="Model A"
)

# Define model
xgbA = XGBRegressor(**xgb_params)

# 5-fold cross-validation
cv_results_A = evaluate_cv(
    xgbA,
    X_train_A,
    y_train_A,
    cv,
    model_name="Model A"
)

# Fit model
xgbA.fit(X_train_A, y_train_A)

# Predictions
y_pred_A = xgbA.predict(X_test_A)

# Evaluation
results_A = evaluate_regression_model(
    y_test_A,
    y_pred_A,
    model_name="Model A"
)

# Scatter plot
save_actual_vs_predicted_plot(
    y_test_A,
    y_pred_A,
    "Model A XGBoost Regression",
    "model_output/impact_score_modelA_XGBoost_scatterplot_existing_predictors.png"
)

# SHAP
save_shap_summary(
    model=xgbA,
    X=X_train_A,
    title="Model A SHAP Values",
    filepath="model_output/impact_score_modelA_XGBoost_existing_predictors_SHAP.png"
)

# ==============================================================================
# MODEL B: LATENT CYBER OPERATION TOPICS
# ==============================================================================
# One-hot encode control variables
X_controls = pd.get_dummies(
    filtered_topic[control_features],
    drop_first=True
)

# One-hot encode BERTopic assignments
X_topics = pd.get_dummies(
    filtered_topic["topic"],
    prefix="topic",
    drop_first=True
)

# Combine controls and topics
X_topic = pd.concat(
    [X_controls, X_topics],
    axis=1
)

# Clean column names for XGBoost
X_topic = clean_column_names(X_topic)

# Encode y labels
y = filtered_orig["impact_indicator_score"]

# Train-test split
X_train_B, X_test_B, y_train_B, y_test_B = split_train_test(
    X_topic,
    y,
    model_name="Model B"
)

# Define model
xgbB = XGBRegressor(**xgb_params)

# 5-fold cross-validation
cv_results_A = evaluate_cv(
    xgbB,
    X_train_B,
    y_train_B,
    cv,
    model_name="Model B"
)

# Fit model
xgbB.fit(X_train_B, y_train_B)

# Predictions
y_pred_B = xgbB.predict(X_test_B)

# Evaluation
results_B = evaluate_regression_model(
    y_test_B,
    y_pred_B,
    model_name="Model B"
)

# Scatter plot
save_actual_vs_predicted_plot(
    y_test_B,
    y_pred_B,
    "Model B XGBoost Regression",
    "model_output/impact_score_modelB_XGBoost_scatterplot_existing_predictors.png"
)

# SHAP
topic_rename = {
    "topic_1_0": "Malware & Cyber Espionage",
    "topic_2_0": "Ransomware & Cyber Extortion",
    "topic_3_0": "DDoS & Service Disruption Campaigns",
    "topic_4_0": "Data Breaches & Information Exposure",
    "topic_5_0": "Institutional Ransomware Attacks",
    "topic_6_0": "Vulnerability Exploitation & Theft",
    "topic_7_0": "Surveillance & Mobile Espionage",
    "topic_8_0": "Government Intelligence Operations"
}

save_shap_summary(
    model=xgbB,
    X=X_train_B,
    title="Model B SHAP Values",
    filepath="model_output/impact_score_XGBoost_latent_topic_predictors_SHAP.png",
    rename_dict=topic_rename
)