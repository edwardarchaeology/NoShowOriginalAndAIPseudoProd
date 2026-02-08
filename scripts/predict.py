"""
Prediction Script for No-Show Model

Loads the trained model and makes predictions on new data.
Adds risk level categories and saves predictions.
"""

import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('scripts')
from feature_engineering import get_feature_columns


def load_model(filepath='models/xgboost_noshow_model.pkl'):
    """Load the trained model."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def load_data(filepath='data/clean/engineered.csv'):
    """Load engineered data for prediction."""
    df = pd.read_csv(filepath)
    return df


def make_predictions(model, X):
    """
    Make predictions with the trained model.
    
    Returns:
        y_pred: Binary predictions (0/1)
        y_pred_proba: Probability of no-show
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return y_pred, y_pred_proba


def assign_risk_levels(probabilities):
    """
    Assign risk level categories based on predicted probability.
    
    Categories:
    - Low: < 0.25
    - Medium: 0.25 - 0.50
    - High: 0.50 - 0.75
    - Very High: >= 0.75
    """
    risk_levels = []
    
    for prob in probabilities:
        if prob < 0.25:
            risk_levels.append('Low')
        elif prob < 0.50:
            risk_levels.append('Medium')
        elif prob < 0.75:
            risk_levels.append('High')
        else:
            risk_levels.append('Very High')
    
    return risk_levels


def create_prediction_report(df, y_pred, y_pred_proba, risk_levels):
    """Create a comprehensive prediction report."""
    # Add predictions to dataframe
    df_pred = df.copy()
    df_pred['predicted_noshow'] = y_pred
    df_pred['noshow_probability'] = y_pred_proba
    df_pred['risk_level'] = risk_levels
    
    # Print summary
    print("="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    print(f"\nTotal predictions: {len(df_pred)}")
    print(f"Predicted no-shows: {y_pred.sum()} ({y_pred.sum()/len(df_pred)*100:.1f}%)")
    print(f"Predicted shows: {(1-y_pred).sum()} ({(1-y_pred).sum()/len(df_pred)*100:.1f}%)")
    
    print("\nRisk Level Distribution:")
    risk_counts = df_pred['risk_level'].value_counts()
    for level in ['Low', 'Medium', 'High', 'Very High']:
        if level in risk_counts.index:
            count = risk_counts[level]
            pct = count / len(df_pred) * 100
            print(f"  {level:12s}: {count:6d} ({pct:5.1f}%)")
    
    print("\nProbability Statistics:")
    print(f"  Mean:   {y_pred_proba.mean():.4f}")
    print(f"  Median: {np.median(y_pred_proba):.4f}")
    print(f"  Min:    {y_pred_proba.min():.4f}")
    print(f"  Max:    {y_pred_proba.max():.4f}")
    
    # If we have actual labels, calculate accuracy
    if 'no_show_binary' in df.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = df['no_show_binary'].values
        
        print("\nModel Performance:")
        print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
        print(f"  F1 Score:  {f1_score(y_true, y_pred):.4f}")
        
        # Risk level accuracy
        print("\nActual No-Show Rate by Predicted Risk Level:")
        for level in ['Low', 'Medium', 'High', 'Very High']:
            mask = df_pred['risk_level'] == level
            if mask.sum() > 0:
                actual_rate = y_true[mask].mean()
                print(f"  {level:12s}: {actual_rate*100:5.1f}% (n={mask.sum()})")
    
    return df_pred


def save_predictions(df_pred, filepath='reports/predictions.csv'):
    """Save predictions to CSV file."""
    # Select key columns for output
    output_columns = [
        'PatientId',
        'AppointmentID',
        'AppointmentDay',
        'ScheduledDay',
        'lead_time_days',
        'is_same_day_or_past',
        'Age',
        'Gender',
        'Neighbourhood',
        'previous_noshow_rate',
        'is_chronic_noshow',
        'noshow_probability',
        'predicted_noshow',
        'risk_level'
    ]
    
    # Include actual if available
    if 'no_show_binary' in df_pred.columns:
        output_columns.append('no_show_binary')
    
    # Filter to available columns
    available_columns = [col for col in output_columns if col in df_pred.columns]
    
    df_output = df_pred[available_columns].copy()
    df_output.to_csv(filepath, index=False)
    
    print(f"\nPredictions saved to {filepath}")


def identify_high_risk_appointments(df_pred, top_n=100):
    """Identify the highest risk appointments."""
    print("\n" + "="*60)
    print(f"TOP {top_n} HIGHEST RISK APPOINTMENTS")
    print("="*60)
    
    # Sort by probability
    high_risk = df_pred.nlargest(top_n, 'noshow_probability')
    
    print("\nCharacteristics of High-Risk Appointments:")
    print(f"  Average probability: {high_risk['noshow_probability'].mean():.4f}")
    print(f"  Average lead time: {high_risk['lead_time_abs'].mean():.1f} days")
    print(f"  Same-day appointments: {high_risk['is_same_day_or_past'].sum()} ({high_risk['is_same_day_or_past'].sum()/len(high_risk)*100:.1f}%)")
    print(f"  Chronic no-showers: {high_risk['is_chronic_noshow'].sum()} ({high_risk['is_chronic_noshow'].sum()/len(high_risk)*100:.1f}%)")
    print(f"  Average age: {high_risk['Age'].mean():.1f}")
    print(f"  SMS sent: {high_risk['SMS_received'].sum()} ({high_risk['SMS_received'].sum()/len(high_risk)*100:.1f}%)")
    
    # Save high-risk appointments
    high_risk.to_csv('reports/high_risk_appointments.csv', index=False)
    print(f"\nTop {top_n} high-risk appointments saved to reports/high_risk_appointments.csv")
    
    return high_risk


def main():
    """Main prediction pipeline."""
    print("="*60)
    print("NO-SHOW PREDICTION - INFERENCE")
    print("="*60)
    
    # 1. Load model
    print("\n1. Loading trained model...")
    model = load_model()
    
    # 2. Load data
    print("\n2. Loading data...")
    df = load_data()
    features = get_feature_columns()
    X = df[features].copy()
    
    print(f"Data shape: {X.shape}")
    
    # 3. Make predictions
    print("\n3. Making predictions...")
    y_pred, y_pred_proba = make_predictions(model, X)
    
    # 4. Assign risk levels
    print("\n4. Assigning risk levels...")
    risk_levels = assign_risk_levels(y_pred_proba)
    
    # 5. Create prediction report
    print("\n5. Creating prediction report...")
    df_pred = create_prediction_report(df, y_pred, y_pred_proba, risk_levels)
    
    # 6. Save predictions
    print("\n6. Saving predictions...")
    save_predictions(df_pred)
    
    # 7. Identify high-risk appointments
    print("\n7. Identifying high-risk appointments...")
    high_risk = identify_high_risk_appointments(df_pred, top_n=100)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
