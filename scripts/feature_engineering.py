"""
Feature Engineering for No-Show Prediction Model

This module creates all features for the XGBoost model, including:
- Temporal features (day_of_week, month, hour, is_weekend)
- Lead time handling (absolute value, bins, same-day flag)
- Patient history features (previous_noshow_rate, is_chronic_noshow, etc.)
- Medical complexity score
- Label encoding for categorical variables
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_raw_data(filepath):
    """Load the raw dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def create_temporal_features(df):
    """Extract temporal features from ScheduledDay and AppointmentDay."""
    # Convert to datetime
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    # Extract features from AppointmentDay
    df['day_of_week'] = df['AppointmentDay'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['AppointmentDay'].dt.month
    df['hour'] = df['ScheduledDay'].dt.hour
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Saturday or Sunday
    
    return df


def create_lead_time_features(df):
    """
    Create lead time features with proper handling of same-day and negative values.
    
    Key insight: Same-day appointments have the LOWEST no-show rate (4.7%),
    while appointments 2-4 weeks out have the HIGHEST rate (32.6%).
    """
    # Calculate lead time in days
    df['lead_time_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.total_seconds() / (24 * 3600)
    
    # Create same-day or past flag (CRITICAL FEATURE - ~61% importance)
    df['is_same_day_or_past'] = (df['lead_time_days'] <= 0).astype(int)
    
    # Get absolute lead time for binning
    df['lead_time_abs'] = df['lead_time_days'].abs()
    
    # Create lead time categories (captures U-shaped relationship)
    def categorize_lead_time(days):
        abs_days = abs(days)
        if days <= 0:
            return 'same_day_or_past'
        elif abs_days <= 7:
            return '1_week'
        elif abs_days <= 14:
            return '2_weeks'
        elif abs_days <= 30:
            return '3_4_weeks'
        else:
            return 'over_month'
    
    df['lead_time_category'] = df['lead_time_days'].apply(categorize_lead_time)
    
    # Label encode the categories
    le_lead_time = LabelEncoder()
    df['lead_time_category_encoded'] = le_lead_time.fit_transform(df['lead_time_category'])
    
    return df


def create_patient_history_features(df):
    """
    Create patient history features using proper time-ordering.
    
    CRITICAL: These features add ~9% to model importance.
    Must sort by PatientId and AppointmentDay, and use cumulative stats
    EXCLUDING the current appointment.
    """
    # Convert No-show to binary (Yes=1, No=0)
    df['no_show_binary'] = (df['No-show'] == 'Yes').astype(int)
    
    # Sort by patient and appointment date
    df = df.sort_values(['PatientId', 'AppointmentDay']).reset_index(drop=True)
    
    # Calculate cumulative statistics EXCLUDING current row
    df['previous_appointments'] = df.groupby('PatientId').cumcount()  # Number of previous appointments
    df['previous_noshows'] = df.groupby('PatientId')['no_show_binary'].cumsum() - df['no_show_binary']
    
    # Calculate previous no-show rate (handle division by zero)
    df['previous_noshow_rate'] = np.where(
        df['previous_appointments'] > 0,
        df['previous_noshows'] / df['previous_appointments'],
        0  # First appointment gets 0
    )
    
    # Flag for first appointment
    df['is_first_appointment'] = (df['previous_appointments'] == 0).astype(int)
    
    # Flag for chronic no-showers (3+ appointments and 50%+ no-show rate)
    df['is_chronic_noshow'] = (
        (df['previous_appointments'] >= 3) & 
        (df['previous_noshow_rate'] >= 0.5)
    ).astype(int)
    
    return df


def create_medical_complexity_score(df):
    """
    Create a medical complexity score by summing medical conditions.
    """
    df['medical_complexity'] = (
        df['Hipertension'] + 
        df['Diabetes'] + 
        df['Alcoholism'] + 
        df['Handcap']
    )
    
    return df


def encode_categorical_features(df):
    """
    Label encode categorical features.
    
    XGBoost handles label encoding natively, so we use this instead of one-hot encoding.
    """
    # Encode Gender (Female=0, Male=1 or similar)
    le_gender = LabelEncoder()
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    
    # Encode Neighbourhood (81 unique values)
    le_neighbourhood = LabelEncoder()
    df['Neighbourhood_encoded'] = le_neighbourhood.fit_transform(df['Neighbourhood'])
    
    return df


def engineer_features(df):
    """
    Main feature engineering pipeline.
    
    Returns:
        DataFrame with all engineered features
    """
    print("Starting feature engineering...")
    print(f"Initial dataset shape: {df.shape}")
    
    # Create temporal features
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    # Create lead time features
    print("Creating lead time features...")
    df = create_lead_time_features(df)
    
    # Create patient history features (MUST be done with proper sorting)
    print("Creating patient history features...")
    df = create_patient_history_features(df)
    
    # Create medical complexity score
    print("Creating medical complexity score...")
    df = create_medical_complexity_score(df)
    
    # Encode categorical features
    print("Encoding categorical features...")
    df = encode_categorical_features(df)
    
    print(f"Final dataset shape: {df.shape}")
    print("Feature engineering complete!")
    
    return df


def get_feature_columns():
    """
    Return list of feature columns to use for modeling.
    """
    features = [
        # Lead time features (CRITICAL)
        'is_same_day_or_past',
        'lead_time_category_encoded',
        'lead_time_abs',
        
        # Patient history features (HIGHLY PREDICTIVE)
        'previous_noshow_rate',
        'is_chronic_noshow',
        'previous_appointments',
        'is_first_appointment',
        
        # Temporal features
        'day_of_week',
        'month',
        'hour',
        'is_weekend',
        
        # Demographics
        'Age',
        'Gender_encoded',
        'Neighbourhood_encoded',
        
        # Medical conditions
        'Scholarship',
        'Hipertension',
        'Diabetes',
        'Alcoholism',
        'Handcap',
        'medical_complexity',
        
        # SMS reminder
        'SMS_received',
    ]
    
    return features


if __name__ == "__main__":
    # Test the feature engineering pipeline
    print("Testing feature engineering pipeline...")
    
    # Load data
    df = load_raw_data('data/raw/noshowappointments-kagglev2-may-2016.csv')
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Save engineered data
    df_engineered.to_csv('data/clean/engineered.csv', index=False)
    print("Engineered data saved to data/clean/engineered.csv")
    
    # Print feature info
    features = get_feature_columns()
    print(f"\nTotal features: {len(features)}")
    print("Feature list:")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
