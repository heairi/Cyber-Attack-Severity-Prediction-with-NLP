# Cyber Incident Intelligence, Topic Modeling, and Severity Prediction

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Project Components](#project-components)
    - [NLP Topic Modeling Pipeline](#1-nlp-topic-modeling-pipeline)
    - [Exploratory Data Analysis](#2-exploratory-data-analysis)
    - [Machine Learning: Predicting Cyber Incident Severity](#3-machine-learning:-predicting-cyber-incident-        severity)



## Overview

This project demonstrates an end-to-end data science workflow that combines Natural Language Processing (NLP), unsupervised learning, statistical analysis, explainable AI, and machine learning to analyze global cyber conflict.

Using incident descriptions from the EuRepoC (European Repository of Cyber Incidents) dataset, I developed a BERTopic-based pipeline to automatically discover latent cyber operation types and investigated how these operational patterns relate to:

- Incident severity
- Political responses
- Geopolitical targeting behavior
- Temporal trends in cyber conflict

I then built and compared multiple XGBoost regression models to predict cyber incident impact from both structured cyber conflict variables and NLP-derived features.

This project highlights skills directly applicable to Data Scientist, Machine Learning Engineer, and Applied NLP roles.

## Business Problem

Can transformer-based topic modeling extract meaningful operational categories from cyber incident descriptions, and can those latent topics be used to understand and predict incident impact?

## Project Components

### 1. NLP Topic Modeling Pipeline

Notebook: `build_BERTopic_model.ipynb`

Raw Incident Descriptions
→ Text Cleaning
→ Geographic Entity Removal (spaCy)
→ Sentence Embeddings (MiniLM)
→ UMAP Dimensionality Reduction
→ BERTopic Topic Modeling
→ Topic Reduction
→ Human-Labeled Cyber Operation Categories

### Extracted Cyber Operation Types

- Website Defacement & Hacktivism
- Malware & Cyber Espionage
- Ransomware & Cyber Extortion
- DDoS & Service Disruption Campaigns
- Data Breaches & Information Exposure
- Institutional Ransomware Attacks
- Vulnerability Exploitation & Theft
- Surveillance & Mobile Espionage
- Government Intelligence Operations

### 2. Exploratory Data Analysis

Notebook: `exploratory_cyber_operations_analysis.ipynb`

Research Questions:

1. What operational patterns connect initiator countries, cyber operation types, and receiver countries?
2. Which cyber operation categories are associated with the most severe incidents?
3. What types of cyber operations provoke political responses?
4. How have cyber operations evolved over time?
5. Which countries are targeted by which types of cyber operations?
6. Are there patterns in the types of cyber attacks used by initiator countries?

Key analytical methods:

- Sankey diagrams
- Heatmaps
- Statistical testing (Kruskal-Wallis, Dunn post-hoc)
- Spearman trend analysis
- Time-series visualizations
- Geographic targeting analysis

Key findings:

- Financially motivated operations are the most common form of cyber conflict.
- Data exposure and ransomware incidents generate the greatest societal impacts.
- Disruption operations provoke the strongest political responses.
- State-linked actors are strongly associated with intrusion and espionage operations.
- The United States is the most frequent recipient of cyber incidents.
- Cyber conflict is becoming increasingly severe and financially motivated over time.

### 3. Machine Learning: Predicting Cyber Incident Severity

Script: `cyber_incident_ml.py`

Target Variable:

`impact_indicator_score`

Composite severity metric:

- Economic Impact
- Political Impact
- Intelligence Impact
- Functional Impact

### Model A: Structured Features

Features:

- Existing cyber operation categories
- Initiator country
- Receiver country
- Conflict variables
- Affected entities

### Model B: NLP-Derived Features

Features:

- BERTopic latent cyber operation categories
- Control variables

### Model C: Combined Features

Features:

- Existing cyber operation categories
- BERTopic features
- Control variables

## Machine Learning Framework

Model:

- XGBoost Regressor

Validation:

- 80/20 Train-Test Split
- 5-Fold Cross Validation

Metrics:

- MAE
- RMSE
- R²

Explainability:

- SHAP
- Feature importance analysis

## Results

| Model | MAE | RMSE | R² |
|---------|---------|---------|---------|
| Model A (Structured Features) | 1.052 | 1.876 | 0.760 |
| Model C (Combined Features) | 1.105 | 1.987 | 0.731 |
| Model B (BERTopic Features) | 1.336 | 2.348 | 0.625 |

### Interpretation

The strongest model was the structured baseline, explaining 76% of variance in incident impact.

The most interesting finding is that BERTopic-derived features alone explained 62.5% of variation in cyber incident severity. This demonstrates that latent representations extracted from unstructured incident descriptions contain substantial predictive information.

## Explainable AI 

SHAP was used to:

- Identify influential predictors
- Explain severity predictions
- Compare structured and NLP-derived feature importance

Generated outputs include:

- SHAP summary plots
- Actual vs predicted diagnostics
- Model comparison visualizations

## Project Structure
project/

├── data/
    ├── eurepoc_dataset
    ├── topic model 
├── model_output/
│   ├── shap
│   ├── figures
│   ├── models
├── notebooks/
│   ├── build_BERTopic_model.ipynb
│   ├── cyber_incident_eda.ipynb

├── src/
│   ├── nlp_functions.py
│   └── ml_functions.py

├── cyber_incident_ml.py
├── requirements.txt
└── README.md

## Key Findings
BERTopic-derived cyber operation categories explained 62.5% of variance in cyber incident severity using NLP features alone.
Structured cyber conflict variables achieved an R² of 0.76 when predicting impact scores.
Financially motivated operations and data exposure incidents generated the highest societal impacts.
Disruption operations triggered the strongest political responses.
State-linked actors were strongly associated with intrusion and cyber espionage campaigns.
Cyber conflict has become increasingly dominated by financially motivated operations over time.
SHAP analysis identified the operational characteristics most strongly associated with severe cyber incidents.

## Why Recruiters Should Care

This project demonstrates the ability to:

- Build production-style NLP pipelines
- Transform unstructured text into predictive features
- Apply machine learning to a real-world cybersecurity problem
- Perform statistical analysis alongside predictive modeling
- Explain model behavior using SHAP
- Communicate findings through data storytelling and visualization

The project spans the full data science lifecycle:

Data Collection → NLP Feature Engineering → Statistical Analysis → Machine Learning → Explainability → Business Interpretation

## Technical Skills Demonstrated

### Machine Learning

- XGBoost
- Regression Modeling
- Cross Validation
- Feature Engineering
- Model Evaluation

### Natural Language Processing

- BERTopic
- Sentence Transformers
- UMAP
- KeyBERT
- spaCy

### Statistics

- Kruskal-Wallis Tests
- Dunn Post-hoc Tests
- Spearman Correlations
- Multiple Testing Correction

### Explainable AI Skills

- SHAP
- Feature Importance Analysis

### Data Visualization

- Plotly
- Sankey Diagrams
- Heatmaps
- Time-Series Analysis
- Seaborn
- Matplotlib

### Software Engineering

- Modular Python Code
- Reusable NLP Pipelines
- Reproducible Research Workflows
- Model Serialization

## Author

Heather Iriye

Data Science | NLP | Machine Learning | Cybersecurity Analytics
