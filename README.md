# Cyber Incident Intelligence, Topic Modeling, and Severity Prediction

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Why Recruiters Should Care](#why-recruiters-should-care)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Project Components](#project-components)
    - [NLP Topic Modeling Pipeline](#1-nlp-topic-modeling-pipeline)
    - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis)
    - [Machine Learning: Predicting Cyber Incident Severity](#3-machine-learning-predicting-cyber-incident-severity)
- [Key Findings](#key-findings)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)
    - [Machine Learning](#machine-learning)
    - [Natural Language Processing](#natural-language-processing)
    - [Software Engineering](#software-engineering)
    - [Explainable AI Skills](#explainable-ai-skills)
    - [Statistics](#statistics)
    - [Data Visualization](#data-visualization)
 - [Author](#author)

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

## Getting Started
The project was developed in **Visual Studio Code 1.135.0** using **Python 3.14.6**. If you're running the project in **GitHub Codespaces**, skip to steps 2 & 3. To run the project outside of codespaces: 

### 1. Clone the repository

```bash
git clone https://github.com/heairi/eurepoc_cyber_security.git
cd eurepoc_cyber_security
```
### 2. Create and activate a virtual environment

Linux or macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` reproduces the full development environment and includes packages used by exploratory notebooks and alternative models. 

## Project Structure

```text
project/
├── data/
│   ├── eurepoc_dataset/                # Processed dataset with latent topics
│   └── topic_model/                    # BERTopic model, topic assignments, and topic info
│
├── model_output/
│   ├── shap/                           # SHAP plots for Models A-C
│   ├── figures/                        # Regression scatterplots & model comparison stacked bar chart
│   └── models/                         # Pickled Models A-C 
│
├── notebooks/
│   ├── build_BERTopic_model.ipynb      # NLP topic modeling 
│   └── cyber_incident_eda.ipynb        # Links between latent cyber operation topics and societal impact
│
├── visualizations/                    # Sankey plot, heatmaps, area plots, line plots, and box plots from EDA
│
├── src/
│   ├── nlp_functions.py                # Functions used to perform NLP topic modeling
│   └── ml_functions.py                 # Functions used to run machine learning pipeline
│
├── cyber_incident_ml.py                # Machine learning pipeline
├── requirements.txt                    # Full project Python environment
└── README.md                            
```

## How to Run the Project
## NLP Topic Modeling
run notebooks/build_BERTopic_model.ipynb

The notebook will: 
- Download the eurepoc dyadic dataset
- Clean incident descriptions
- Generate sentence embeddings (Sentence-BERT)
- Distill topics into readable topic names and groups
- Save model and topic modelling output
*Note: To obtain the same topic modeling results, the environment must be the exact same as during development (i.e., pip install -r requirements.txt)

## Latent Cyber Operation Topic EDA
run notebooks/cyber_incident_eda.ipynb

Using the latent topics extracted from incident descriptions (see build_BERTopic_model.ipynb), are specific types of cyber attacks linked to differences in terms of societal impact and political responses? 

## Machine Learning
From the repository root, run: 
cyber_incident_ml.py

The script will:
- load the original eurepoc dyadic dataset
- load the processed dataset with the latent cyber operation categories extracted using NLP
- create the feature and target vectors for Models A-C
- train and test Models A-C
- generate regression scatterplots & SHAP feature importance plots
- compare model metrics (MAE, RMSE, R²)
- save model output

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

## Key Findings
BERTopic-derived cyber operation categories explained 62.5% of variance in cyber incident severity using NLP features alone.
Structured cyber conflict variables achieved an R² of 0.76 when predicting impact scores.
Financially motivated operations and data exposure incidents generated the highest societal impacts.
Disruption operations triggered the strongest political responses.
State-linked actors were strongly associated with intrusion and cyber espionage campaigns.
Cyber conflict has become increasingly dominated by financially motivated operations over time.
SHAP analysis identified the operational characteristics most strongly associated with severe cyber incidents.

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

### Software Engineering

- Modular Python Code
- Reusable NLP Pipelines
- Reproducible Research Workflows
- Model Serialization

### Explainable AI Skills

- SHAP
- Feature Importance Analysis

### Statistics

- Kruskal-Wallis Tests
- Dunn Post-hoc Tests
- Spearman Correlations
- Multiple Testing Correction

### Data Visualization

- Plotly
- Sankey Diagrams
- Heatmaps
- Time-Series Analysis
- Seaborn
- Matplotlib


## Author

**Heather Iriye, PhD** 

Data Science | NLP | Machine Learning | Cybersecurity Analytics
