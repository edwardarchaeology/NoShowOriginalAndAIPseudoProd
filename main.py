"""
Main Training Pipeline for No-Show Prediction Model

This script:
1. Loads and engineers features
2. Performs stratified train/val/test split (64/16/20)
3. Trains XGBoost model with class imbalance handling
4. Evaluates performance (ROC-AUC, PR-AUC, F1, etc.)
5. Generates visualizations
6. Saves the trained model
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score
)
import xgboost as xgb

# Import feature engineering functions
import sys
sys.path.append('scripts')
from feature_engineering import (
    load_raw_data,
    engineer_features,
    get_feature_columns
)


def prepare_data(df, features, target='no_show_binary'):
    """Prepare features and target for modeling."""
    X = df[features].copy()
    y = df[target].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y


def stratified_split(X, y, random_state=42):
    """
    Perform stratified train/val/test split (64/16/20).
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    
    # Second split: 64% train, 16% val (from the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=random_state
    )
    
    print("\nData split:")
    print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model with specified hyperparameters and class imbalance handling.
    
    Expected performance:
    - ROC-AUC: ~0.75
    - PR-AUC: ~0.42
    - F1: ~0.46
    - Recall: ~79%
    """
    # Calculate scale_pos_weight for class imbalance
    # Formula: (number of negative class) / (number of positive class)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    print(f"\nClass imbalance:")
    print(f"Negative samples: {n_neg}")
    print(f"Positive samples: {n_pos}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Define model parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'tree_method': 'hist',  # Faster training
        'enable_categorical': False,  # We use label encoding
    }
    
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    print("Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test, dataset_name='Test'):
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} Set")
    print('='*60)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nPerformance Metrics:")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Show', 'No-Show']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }
    
    return metrics


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTop 10 Features by Importance:")
    for i in range(min(10, top_n)):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:35s} {importance[idx]*100:6.2f}%")


def plot_roc_curve(metrics):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_pred_proba'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(metrics):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(metrics['y_true'], metrics['y_pred_proba'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR (AUC = {metrics['pr_auc']:.4f})")
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Show', 'No-Show'],
                yticklabels=['Show', 'No-Show'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_model(model, filepath='models/xgboost_noshow_model.pkl'):
    """Save trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filepath}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("NO-SHOW PREDICTION MODEL - TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load raw data
    print("\n1. Loading raw data...")
    df = load_raw_data('data/raw/noshowappointments-kagglev2-may-2016.csv')
    print(f"Raw data shape: {df.shape}")
    
    # 2. Engineer features
    print("\n2. Engineering features...")
    df_engineered = engineer_features(df)
    
    # Save engineered data
    df_engineered.to_csv('data/clean/engineered.csv', index=False)
    print("Engineered data saved to data/clean/engineered.csv")
    
    # 3. Prepare data for modeling
    print("\n3. Preparing data for modeling...")
    features = get_feature_columns()
    X, y = prepare_data(df_engineered, features)
    
    # 4. Stratified split
    print("\n4. Performing stratified split...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    
    # 5. Train model
    print("\n5. Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # 6. Evaluate model
    print("\n6. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # 7. Generate visualizations
    print("\n7. Generating visualizations...")
    plot_feature_importance(model, features)
    plot_roc_curve(metrics)
    plot_pr_curve(metrics)
    plot_confusion_matrix(metrics['confusion_matrix'])
    print("Visualizations saved to reports/figures/")
    
    # 8. Save model
    print("\n8. Saving model...")
    save_model(model)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    
    # Final summary
    print("\nFinal Performance Summary:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} (Expected: ~0.75)")
    print(f"PR-AUC:  {metrics['pr_auc']:.4f} (Expected: ~0.42)")
    print(f"F1:      {metrics['f1']:.4f} (Expected: ~0.46)")


if __name__ == "__main__":
    main()
