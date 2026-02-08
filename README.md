# NOTE

This repo was constructed from my original quick investigation script, original_mvp.py, via AI (Github Copilot with Claude Sonnet 4.5). The original script has been included for reference as well as the prompt I used to build out this repo. I performed a quick manual check of the scripts in this AI generated repo to make sure the logic was sound and the output was consistent with my original implementation. The intent was to create a codebase and documentation more similar to what I would build in a production environment rather than a quick EDA + model build for an interview within the 30-45 min timeline I was suggested to stick to.

# No-Show Prediction Model

A comprehensive XGBoost-based machine learning model to predict medical appointment no-shows. This project analyzes ~110K appointments from the Kaggle dataset and achieves **ROC-AUC of 0.75** with full explainability through SHAP analysis.

## 🎯 Project Overview

**Goal**: Predict which medical appointments are likely to result in no-shows and understand the factors driving this behavior.

**Key Achievement**: Discovered the **U-shaped relationship** between lead time and no-shows:

- Same-day appointments: **4.7% no-show rate** (LOWEST risk - protective)
- 2-4 weeks out: **32.6% no-show rate** (HIGHEST risk - 7x higher than same-day)

## 📊 Model Performance

| Metric                | Value  | Target |
| --------------------- | ------ | ------ |
| **ROC-AUC**           | 0.7498 | ~0.75  |
| **PR-AUC**            | 0.4154 | ~0.42  |
| **F1 Score**          | 0.4549 | ~0.46  |
| **Recall (No-Shows)** | 78%    | ~79%   |

## 🔍 Key Insights Discovered

### 1. Lead Time Has U-Shaped Relationship (CRITICAL) ✅

- Same-day appointments have the **LOWEST** no-show rate (4.7%)
- Appointments 2-4 weeks out have the **HIGHEST** rate (32.6%)
- This pattern is captured by the `is_same_day_or_past` feature (71% model importance)

### 2. Chronic No-Showers Drive Impact ✅

- **2.7% of patients** are chronic no-showers (3+ appointments, 50%+ no-show rate)
- They cause **20.1% of all no-shows**
- First appointment behavior predicts future: 30.8% future no-show if missed first vs 19.3% if showed

### 3. SMS Paradox Explained ✅

- SMS reminders sent only for appointments 3+ days out (threshold-based policy)
- 0% of same-day appointments get SMS, 60% of 4+ day appointments do
- SMS correlates with no-shows because it's sent to inherently riskier appointments (long lead time)

### 4. Age Pattern ✅

- Negative correlation: younger patients (11-20) have 25% no-show rate
- Older patients (60+) have 15% no-show rate

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12
- uv package manager

### Installation

```bash
# Clone the repository
cd NoShowModel

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install xgboost scikit-learn pandas numpy matplotlib seaborn kaggle shap

# Download data from Kaggle (requires Kaggle API credentials)
# The data is automatically downloaded when running main.py
```

## 📁 Project Structure

```
NoShowModel/
├── main.py                          # Main training pipeline
├── data/
│   ├── raw/                         # Kaggle dataset (downloaded)
│   └── clean/                       # Engineered features
├── models/
│   └── xgboost_noshow_model.pkl    # Trained model
├── scripts/
│   ├── feature_engineering.py       # Feature creation
│   ├── shap_analysis.py            # Model explainability
│   ├── predict.py                  # Inference script
│   ├── analyze_lead_time.py        # Lead time analysis
│   ├── analyze_patient_patterns.py # Patient behavior analysis
│   ├── analyze_age.py              # Age pattern analysis
│   └── analyze_neighborhoods.py    # Neighbourhood analysis
└── reports/
    ├── figures/                     # Visualizations
    ├── predictions.csv             # Model predictions
    ├── high_risk_predictions.csv   # High-risk appointments
    └── neighbourhood_stats.csv     # Neighbourhood statistics
```

## 🎓 Usage

### 1. Train the Model

```bash
python main.py
```

This script:

- Downloads data from Kaggle (if not already present)
- Engineers 21 features including patient history and lead time patterns
- Trains XGBoost with class imbalance handling (scale_pos_weight=3.95)
- Evaluates on stratified 64/16/20 train/val/test split
- Generates performance visualizations
- Saves trained model to `models/`

### 2. Run SHAP Analysis

```bash
python scripts/shap_analysis.py
```

Generates:

- Global feature importance plots
- Individual prediction explanations (waterfall plots)
- Dependence plots for top features
- High-risk pattern analysis

### 3. Analyze Patterns

```bash
# Lead time analysis (U-shaped curve)
python scripts/analyze_lead_time.py

# Patient behavior analysis
python scripts/analyze_patient_patterns.py

# Age pattern analysis
python scripts/analyze_age.py

# Neighbourhood analysis
python scripts/analyze_neighborhoods.py
```

### 4. Make Predictions

```bash
python scripts/predict.py
```

Generates predictions with risk levels:

- **Low**: < 25% probability
- **Medium**: 25-50% probability
- **High**: 50-75% probability
- **Very High**: ≥ 75% probability

## 🔧 Features Engineered (21 Total)

### Lead Time Features (CRITICAL - 77% combined importance)

1. `is_same_day_or_past` - Binary flag for same-day or past appointments (71% importance)
2. `lead_time_category_encoded` - Categorical lead time bins (5% importance)
3. `lead_time_abs` - Absolute lead time in days (1% importance)

### Patient History Features (Highly Predictive - 9% importance)

4. `previous_noshow_rate` - Patient's historical no-show percentage (2% importance)
5. `is_chronic_noshow` - Flag for chronic no-showers (7% importance)
6. `previous_appointments` - Total prior appointments
7. `is_first_appointment` - Binary flag for first-time patients (1% importance)

### Temporal Features

8. `day_of_week` - Day of the week (0=Monday, 6=Sunday)
9. `month` - Month of the year
10. `hour` - Hour of scheduling
11. `is_weekend` - Weekend flag

### Demographics

12. `Age` - Patient age (1% importance)
13. `Gender_encoded` - Gender (label encoded)
14. `Neighbourhood_encoded` - Neighbourhood (label encoded, 81 unique)

### Medical Conditions

15. `Scholarship` - Welfare program enrollment
16. `Hipertension` - Hypertension flag
17. `Diabetes` - Diabetes flag
18. `Alcoholism` - Alcoholism flag (1% importance)
19. `Handcap` - Handicap level
20. `medical_complexity` - Sum of medical conditions

### Communication

21. `SMS_received` - SMS reminder sent (1% importance)

## 📈 Visualizations Generated

All visualizations are saved to `reports/figures/`:

1. **Feature Importance** - Top 20 features by importance
2. **ROC Curve** - Model discrimination ability
3. **Precision-Recall Curve** - Performance on imbalanced data
4. **Confusion Matrix** - Classification results
5. **SHAP Summary Plot** - Feature impact on predictions
6. **SHAP Bar Plot** - Mean absolute SHAP values
7. **SHAP Waterfall Plots** - Individual prediction explanations
8. **SHAP Dependence Plots** - Feature relationships (top 5 features)
9. **Lead Time U-Curve** - Critical visualization showing U-shaped relationship
10. **SMS vs Lead Time** - SMS paradox explanation
11. **Patient Distributions** - Patient behavior patterns
12. **Chronic No-Show Impact** - Impact visualization
13. **Age Analysis** - Age vs no-show patterns
14. **Age-Gender Interaction** - Combined age and gender effects
15. **Neighbourhood Analysis** - Neighbourhood variation
16. **Top Neighbourhoods Comparison** - High-volume neighbourhoods

## 🧪 Verification of Statistics

All statistics from the original analysis have been verified:

| Statistic                          | Expected | Actual | ✓   |
| ---------------------------------- | -------- | ------ | --- |
| Same-day no-show rate              | ~4.7%    | 4.7%   | ✅  |
| Same-day appointments              | ~35%     | 34.9%  | ✅  |
| Chronic no-showers (% of patients) | 2.7%     | 2.7%   | ✅  |
| No-shows from chronic patients     | 20.1%    | 20.1%  | ✅  |
| Future no-show if showed first     | ~19%     | 19.3%  | ✅  |
| Future no-show if missed first     | ~32%     | 30.8%  | ✅  |
| ROC-AUC                            | ~0.75    | 0.7498 | ✅  |
| PR-AUC                             | ~0.42    | 0.4154 | ✅  |
| F1 Score                           | ~0.46    | 0.4549 | ✅  |

## 🏆 Top 10 Features by Importance

1. `is_same_day_or_past` - 71.27%
2. `is_chronic_noshow` - 7.10%
3. `lead_time_category_encoded` - 4.73%
4. `previous_noshow_rate` - 2.09%
5. `lead_time_abs` - 1.30%
6. `Age` - 1.28%
7. `SMS_received` - 1.12%
8. `Alcoholism` - 0.98%
9. `is_first_appointment` - 0.93%
10. `Scholarship` - 0.92%

## 🎯 Model Configuration

```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=3.95,  # Handle 4:1 class imbalance
    max_depth=5,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42
)
```

## 💡 Key Takeaways

1. **Don't assume same-day = high risk** - It's actually the lowest!
2. **Patient history is gold** - 9% of model importance
3. **SMS correlation ≠ causation** - Confounded by lead time policy
4. **Label encoding > One-hot encoding** - For XGBoost with categorical variables
5. **Class imbalance matters** - Use `scale_pos_weight` for 4:1 imbalance
6. **Explainability is crucial** - SHAP analysis reveals actionable insights

## 📝 License

This project uses the [No-Show Appointments dataset](https://www.kaggle.com/datasets/raniahelmy/no-show-investigate-dataset) from Kaggle.

## 🙏 Acknowledgments

- Dataset: Kaggle - raniahelmy/no-show-investigate-dataset
- XGBoost library for efficient gradient boosting
- SHAP library for model explainability
- Python scientific computing stack (numpy, pandas, scikit-learn)

---

**Built with Python 3.12, XGBoost, and ❤️**
