"""
Minimal No-Show Prediction Model - MVP Script
Loads data, engineers features, trains XGBoost, and prints scores.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
import xgboost as xgb

def load_and_preprocess():
    """Load and preprocess the data with minimal feature engineering."""
    print("Loading data...")
    df = pd.read_csv('data/raw/noshowappointments-kagglev2-may-2016.csv')
    
    # Convert dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    # Sort by patient and date for history features
    df = df.sort_values(['PatientId', 'ScheduledDay'])
    
    # Target variable
    df['no_show_binary'] = (df['No-show'] == 'Yes').astype(int)
    
    # Lead time features
    df['lead_time_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.total_seconds() / 86400
    df['is_same_day'] = (df['lead_time_days'] == 0).astype(int)
    
    # Patient history features
    df['prev_appointments'] = df.groupby('PatientId').cumcount()
    df['prev_noshows'] = df.groupby('PatientId')['no_show_binary'].shift(1).fillna(0).groupby(df['PatientId']).cumsum()
    df['previous_noshow_rate'] = np.where(
        df['prev_appointments'] > 0,
        df['prev_noshows'] / df['prev_appointments'],
        0
    )
    
    # Temporal features
    df['scheduled_day_of_week'] = df['ScheduledDay'].dt.dayofweek
    df['scheduled_hour'] = df['ScheduledDay'].dt.hour
    df['appointment_day_of_week'] = df['AppointmentDay'].dt.dayofweek
    
    # Encode categoricals
    df['Gender_encoded'] = (df['Gender'] == 'M').astype(int)
    df['Neighbourhood_encoded'] = df['Neighbourhood'].astype('category').cat.codes
    
    # Feature list
    features = [
        'Age', 'Gender_encoded', 'Scholarship', 'Hipertension', 'Diabetes',
        'Alcoholism', 'Handcap', 'SMS_received', 'Neighbourhood_encoded',
        'lead_time_days', 'is_same_day', 'prev_appointments', 'previous_noshow_rate',
        'scheduled_day_of_week', 'scheduled_hour', 'appointment_day_of_week'
    ]
    
    X = df[features]
    y = df['no_show_binary']
    
    print(f"Dataset: {len(df)} appointments, {X.shape[1]} features")
    print(f"No-show rate: {y.mean()*100:.2f}%")
    
    return X, y

def train_and_evaluate(X, y):
    """Train XGBoost model and evaluate."""
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Calculate class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print(f"\nTraining XGBoost model...")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Class imbalance weight: {scale_pos_weight:.2f}")
    
    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        tree_method='hist',
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\nROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Show', 'No-Show']))
    
    return model

def main():
    """Main pipeline."""
    print("="*60)
    print("MINIMAL NO-SHOW PREDICTION MODEL")
    print("="*60 + "\n")
    
    # Load and preprocess
    X, y = load_and_preprocess()
    
    # Train and evaluate
    model = train_and_evaluate(X, y)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
