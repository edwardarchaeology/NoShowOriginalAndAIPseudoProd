"""
SHAP Analysis for No-Show Prediction Model

This script provides comprehensive model explainability using SHAP values:
- Global feature importance
- Individual prediction explanations
- Dependence plots for top features
- High-risk pattern analysis
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# Import feature engineering functions
import sys
sys.path.append('scripts')
from feature_engineering import get_feature_columns


def load_model(filepath='models/xgboost_noshow_model.pkl'):
    """Load the trained model."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def load_engineered_data(filepath='data/clean/engineered.csv'):
    """Load the engineered dataset."""
    df = pd.read_csv(filepath)
    return df


def compute_shap_values(model, X, sample_size=5000):
    """
    Compute SHAP values using TreeExplainer.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        sample_size: Number of samples to use (for speed)
    
    Returns:
        explainer, shap_values
    """
    print(f"Computing SHAP values for {min(sample_size, len(X))} samples...")
    
    # Sample data if needed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer(X_sample)
    
    print("SHAP values computed!")
    
    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, feature_names):
    """Create SHAP summary plot (beeswarm)."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Feature Impact on No-Show Predictions', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved to reports/figures/shap_summary_plot.png")


def plot_shap_bar(shap_values, feature_names):
    """Create SHAP bar plot (mean absolute SHAP values)."""
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title('SHAP Feature Importance (Mean |SHAP|)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP bar plot saved to reports/figures/shap_bar_plot.png")


def plot_shap_waterfall(shap_values, X_sample, feature_names, example_idx=0):
    """
    Create SHAP waterfall plot for individual prediction.
    
    Args:
        example_idx: Index of the example to explain
    """
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[example_idx], show=False)
    plt.title(f'SHAP Waterfall Plot - Example Prediction #{example_idx}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'reports/figures/shap_waterfall_example_{example_idx}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP waterfall plot saved for example {example_idx}")


def plot_shap_dependence(shap_values, X_sample, feature_name, feature_names):
    """Create SHAP dependence plot for a specific feature."""
    feature_idx = feature_names.index(feature_name)
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx, 
        shap_values.values, 
        X_sample, 
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot - {feature_name}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'reports/figures/shap_dependence_{feature_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP dependence plot saved for {feature_name}")


def analyze_high_risk_patterns(model, df, features, threshold=0.5):
    """
    Analyze patterns in high-risk predictions.
    
    Args:
        threshold: Probability threshold for high-risk classification
    """
    print(f"\nAnalyzing high-risk patterns (threshold >= {threshold})...")
    
    X = df[features].copy()
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Create high-risk flag
    df['high_risk'] = (y_pred_proba >= threshold).astype(int)
    df['pred_probability'] = y_pred_proba
    
    # Analyze high-risk characteristics
    high_risk_df = df[df['high_risk'] == 1]
    low_risk_df = df[df['high_risk'] == 0]
    
    print(f"\nHigh-risk appointments: {len(high_risk_df)} ({len(high_risk_df)/len(df)*100:.1f}%)")
    print(f"Low-risk appointments: {len(low_risk_df)} ({len(low_risk_df)/len(df)*100:.1f}%)")
    
    # Compare key features
    print("\nHigh-risk vs Low-risk Characteristics:")
    
    comparison_features = [
        'is_same_day_or_past',
        'lead_time_abs',
        'previous_noshow_rate',
        'is_chronic_noshow',
        'Age',
        'SMS_received'
    ]
    
    for feat in comparison_features:
        if feat in df.columns:
            high_mean = high_risk_df[feat].mean()
            low_mean = low_risk_df[feat].mean()
            print(f"  {feat:30s}: High={high_mean:6.2f}  Low={low_mean:6.2f}  Diff={high_mean-low_mean:6.2f}")
    
    # Save high-risk predictions
    high_risk_df.to_csv('reports/high_risk_predictions.csv', index=False)
    print("\nHigh-risk predictions saved to reports/high_risk_predictions.csv")
    
    return df


def create_shap_force_plot_html(explainer, shap_values, X_sample, feature_names, num_examples=5):
    """
    Create interactive SHAP force plots and save as HTML.
    """
    print(f"\nCreating SHAP force plots for {num_examples} examples...")
    
    # Create force plot for multiple examples
    shap.initjs()
    
    # Save force plot as HTML
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values.values[:num_examples], 
        X_sample.iloc[:num_examples],
        feature_names=feature_names
    )
    
    shap.save_html('reports/figures/shap_force_plot.html', force_plot)
    print("SHAP force plot saved to reports/figures/shap_force_plot.html")


def main():
    """Main SHAP analysis pipeline."""
    print("="*60)
    print("SHAP ANALYSIS - MODEL EXPLAINABILITY")
    print("="*60)
    
    # 1. Load model and data
    print("\n1. Loading model and data...")
    model = load_model()
    df = load_engineered_data()
    features = get_feature_columns()
    X = df[features].copy()
    
    print(f"Data shape: {X.shape}")
    
    # 2. Compute SHAP values
    print("\n2. Computing SHAP values...")
    explainer, shap_values, X_sample = compute_shap_values(model, X, sample_size=5000)
    
    # 3. Create global importance plots
    print("\n3. Creating global importance plots...")
    plot_shap_summary(shap_values, X_sample, features)
    plot_shap_bar(shap_values, features)
    
    # 4. Create waterfall plots for individual predictions
    print("\n4. Creating waterfall plots for example predictions...")
    plot_shap_waterfall(shap_values, X_sample, features, example_idx=0)
    plot_shap_waterfall(shap_values, X_sample, features, example_idx=1)
    plot_shap_waterfall(shap_values, X_sample, features, example_idx=100)
    
    # 5. Create dependence plots for top features
    print("\n5. Creating dependence plots for top features...")
    top_features = [
        'is_same_day_or_past',
        'lead_time_category_encoded',
        'previous_noshow_rate',
        'is_chronic_noshow',
        'Age'
    ]
    
    for feat in top_features:
        if feat in features:
            plot_shap_dependence(shap_values, X_sample, feat, features)
    
    # 6. Analyze high-risk patterns
    print("\n6. Analyzing high-risk patterns...")
    df_with_predictions = analyze_high_risk_patterns(model, df, features, threshold=0.5)
    
    # 7. Create interactive force plots
    print("\n7. Creating interactive force plots...")
    try:
        create_shap_force_plot_html(explainer, shap_values, X_sample, features, num_examples=10)
    except Exception as e:
        print(f"Warning: Could not create force plot HTML: {e}")
    
    print("\n" + "="*60)
    print("SHAP ANALYSIS COMPLETE!")
    print("="*60)
    print("\nAll visualizations saved to reports/figures/")


if __name__ == "__main__":
    main()
