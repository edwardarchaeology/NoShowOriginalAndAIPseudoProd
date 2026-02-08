# Prompt: Build XGBoost No-Show Prediction Model

I need to build a comprehensive machine learning model to predict medical appointment no-shows using XGBoost. The dataset is from Kaggle (raniahelmy/no-show-investigate-dataset) with ~110K appointments.

## Project Requirements

### Data Understanding

The dataset contains:

- **Target**: `No-show` column (Yes/No) - 20.2% no-show rate (imbalanced)
- **Features**: PatientId, AppointmentID, Gender, Age, ScheduledDay, AppointmentDay, Neighbourhood (81 unique), Scholarship (welfare), Hipertension, Diabetes, Alcoholism, Handcap, SMS_received
- **Key Issue**: ~35% of appointments have same-day or negative lead times (data quality issue)

### Critical Insights to Implement

1. **Lead Time Has U-Shaped Relationship with No-Shows** (CRITICAL):
   - Same-day appointments: 4.7% no-show (LOWEST risk - protective)
   - 2-4 weeks out: 32.6% no-show (HIGHEST risk - 7x higher than same-day)
   - Create `is_same_day_or_past` flag (will be top predictor at ~60% importance)
   - Use lead time categories, not just raw days

2. **SMS Paradox Explained**:
   - SMS reminders sent only for appointments 3+ days out (threshold-based policy)
   - 0% of same-day appointments get SMS, 60% of 4+ day appointments do
   - SMS correlates with no-shows because it's sent to inherently riskier appointments (long lead time)
   - Model should learn this correctly

3. **Patient History is Highly Predictive**:
   - 2.7% of patients (chronic no-showers) cause 20.1% of all no-shows
   - First appointment predicts future: 31.8% no-show rate if they missed first vs 19.0% if they showed
   - **Must create these features** (adds ~9% model importance):
     - `previous_noshow_rate`: Patient's historical no-show percentage
     - `is_chronic_noshow`: Flag for patients with 3+ appointments and 50%+ no-show rate
     - `previous_appointments`: Total prior appointments
     - `is_first_appointment`: Binary flag (first-time patients have lower risk)
   - Sort by PatientId and AppointmentDay, use cumulative stats EXCLUDING current appointment

4. **Age Pattern**:
   - Negative correlation: younger patients (11-20) have 25% no-show rate
   - Older patients (60+) have 15% no-show rate
   - Age is mildly predictive (~1% importance)

5. **Neighbourhood Variation**:
   - Wide range: 0% to 100% no-show rates
   - Use label encoding (not one-hot) for 81 neighbourhoods
   - Some correlation with lead time patterns

### Feature Engineering Best Practices

**DO:**

- ✅ Use **label encoding** for categorical variables (Gender, Neighbourhood) - XGBoost handles these natively
- ✅ Extract temporal features: day_of_week, month, hour, is_weekend
- ✅ Create medical complexity score (sum of conditions)
- ✅ Handle negative lead times explicitly with `is_same_day_or_past` flag
- ✅ Create lead time categories/bins (non-linear relationship)
- ✅ Engineer patient history features (critical!)
- ✅ Keep features interpretable for SHAP analysis

**DON'T:**

- ❌ One-hot encode Gender (wasteful - just need 0/1)
- ❌ One-hot encode Neighbourhood (80 dummy variables unnecessary)
- ❌ Drop same-day appointments (they're the lowest risk!)
- ❌ Treat lead time as purely linear
- ❌ Ignore patient history

### Model Configuration

**XGBoost Parameters:**

```python
objective='binary:logistic'
eval_metric='auc'
scale_pos_weight=3.95  # Handle 4:1 class imbalance (88208/22319)
max_depth=5
learning_rate=0.05
n_estimators=500
subsample=0.8
colsample_bytree=0.8
min_child_weight=3
random_state=42
```

**Data Split:**

- Stratified train/val/test: 64/16/20 split
- Use same random_state=42 for reproducibility

**Expected Performance:**

- ROC-AUC: ~0.75
- Precision-Recall AUC: ~0.42
- F1: ~0.46
- Recall for no-shows: ~79%

### Explainability Requirements

**Implement SHAP analysis:**

- Global feature importance (summary plot, bar chart)
- Individual prediction explanations (waterfall plots)
- Dependence plots for top features
- Analyze high-risk patterns (>=50% threshold)
- Save SHAP values for later use

This answers the "WHY" question - not just who will no-show, but what factors drive it.

### Analysis Scripts to Create

1. **Feature Engineering** (`scripts/feature_engineering.py`):
   - Temporal features extraction
   - Label encoding for categoricals
   - Patient history features (with proper sorting and cumsum)
   - Lead time handling (absolute value, bins, same-day flag)
   - Medical complexity score

2. **Main Training Pipeline** (`main.py`):
   - Load and engineer features
   - Stratified split
   - XGBoost training with class imbalance handling
   - Comprehensive evaluation (ROC, PR curves, confusion matrix, classification report)
   - Feature importance plots
   - Model persistence (pickle)

3. **SHAP Explainability** (`scripts/shap_analysis.py`):
   - TreeExplainer for XGBoost
   - Summary plots, bar plots, waterfall plots
   - Dependence plots for top features
   - High-risk pattern analysis

4. **Prediction Script** (`scripts/predict.py`):
   - Load model and make predictions
   - Add risk level categories (Low/Medium/High/Very High)
   - Save predictions with probabilities

5. **Analysis Scripts**:
   - `analyze_neighborhoods.py`: No-show rates by neighbourhood
   - `analyze_age.py`: Age vs no-show patterns
   - `analyze_lead_time.py`: Lead time relationships (creates U-shaped curve visualization)
   - `analyze_patient_patterns.py`: Patient-level behavior, chronic no-showers

### Dependencies

```toml
dependencies = [
    "xgboost>=2.0.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "shap>=0.42.0",
]
```

### Key Deliverables

1. **Trained model** saved to `models/xgboost_noshow_model.pkl`
2. **Visualizations** in `reports/figures/`:
   - Feature importance plots
   - ROC and PR curves
   - Confusion matrix
   - SHAP visualizations (summary, waterfall, dependence plots)
   - U-shaped lead time curve
   - Patient pattern analysis
   - Neighbourhood analysis
3. **Engineered dataset** at `data/clean/engineered.csv` (25 features)
4. **Performance metrics** and model evaluation

### Expected Top Features (in order)

1. `is_same_day_or_past` (~61%)
2. `lead_time_category_encoded` (~15%)
3. `is_chronic_noshow` (~6%)
4. `previous_noshow_rate` (~2%)
5. `Age` (~1%)
6. `SMS_received` (~1%)

### Implementation Notes

- Start with data exploration (ydata-profiling report if helpful)
- Handle negative lead times explicitly (don't filter them out)
- Ensure patient history features use proper time-ordering
- Use label encoding, not one-hot encoding
- Focus on interpretability (SHAP analysis)
- Create visualizations for key relationships (U-shaped curve is critical)

### Avoid These Pitfalls

1. Don't assume same-day = high risk (it's actually the lowest!)
2. Don't interpret SMS correlation as causal (confounded by lead time)
3. Don't ignore patient history (9% of model importance)
4. Don't one-hot encode everything (XGBoost doesn't need it)
5. Don't forget class imbalance handling (scale_pos_weight)

### Success Criteria

- Model AUC > 0.74
- Can identify chronic no-showers automatically
- SHAP explanations are interpretable and actionable
- All key insights discovered and implemented
- Comprehensive visualizations created
