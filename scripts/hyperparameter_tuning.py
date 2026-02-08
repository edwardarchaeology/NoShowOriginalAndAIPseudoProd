"""
Unused due to computational constraints.

Hyperparameter Tuning for XGBoost No-Show Model

Uses Optuna for Bayesian optimization to find optimal hyperparameters.
Optimizes for ROC-AUC using stratified cross-validation.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Import feature engineering functions
import sys
sys.path.append('scripts')
from feature_engineering import (
    load_raw_data,
    engineer_features,
    get_feature_columns
)

# Try to import optuna, provide fallback
try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Install with: uv pip install optuna")


def objective(trial, X_train, y_train, scale_pos_weight):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target
        scale_pos_weight: Fixed class imbalance weight
    
    Returns:
        Mean ROC-AUC score from cross-validation
    """
    # Define hyperparameter search space
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,  # Keep fixed for class imbalance
        'tree_method': 'hist',
        'random_state': 42,
        
        # Hyperparameters to tune
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),  # L2 regularization
    }
    
    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        auc_score = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(auc_score)
        
        # Report intermediate value for pruning
        trial.report(auc_score, fold)
        
        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(cv_scores)


def tune_hyperparameters(X_train, y_train, n_trials=100):
    """
    Run hyperparameter tuning using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
    
    Returns:
        best_params: Dictionary of best hyperparameters
        study: Optuna study object
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter tuning. "
                         "Install with: uv pip install optuna")
    
    # Calculate scale_pos_weight (keep fixed)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    print(f"Using 5-fold cross-validation")
    print(f"Fixed scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize ROC-AUC
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
        study_name='xgboost_noshow_optimization'
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['tree_method'] = 'hist'
    best_params['random_state'] = 42
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest ROC-AUC: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for param, value in sorted(best_params.items()):
        print(f"  {param:20s}: {value}")
    
    return best_params, study


def save_best_params(best_params, filepath='models/best_hyperparameters.json'):
    """Save best hyperparameters to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest hyperparameters saved to {filepath}")


def plot_optimization_history(study):
    """Plot optimization history and parameter importances."""
    try:
        import matplotlib.pyplot as plt
        
        # Optimization history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Optimization history
        trials = study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        
        axes[0].plot(values, alpha=0.6, label='Trial AUC')
        axes[0].plot(np.maximum.accumulate(values), 'r-', linewidth=2, label='Best AUC')
        axes[0].set_xlabel('Trial', fontsize=12)
        axes[0].set_ylabel('ROC-AUC', fontsize=12)
        axes[0].set_title('Optimization History', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Parameter importances
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        values = list(importances.values())
        
        axes[1].barh(params, values, color='steelblue')
        axes[1].set_xlabel('Importance', fontsize=12)
        axes[1].set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/figures/hyperparameter_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Optimization plots saved to reports/figures/hyperparameter_optimization.png")
    except ImportError:
        print("Matplotlib not available for plotting")


def compare_with_baseline(best_params, X_train, y_train, X_test, y_test):
    """
    Compare optimized model with baseline parameters.
    
    Args:
        best_params: Optimized hyperparameters
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        Comparison results
    """
    print("\n" + "="*60)
    print("COMPARING OPTIMIZED VS BASELINE MODEL")
    print("="*60)
    
    # Baseline parameters (from main.py)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    baseline_params = {
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
        'tree_method': 'hist',
    }
    
    # Train baseline model
    print("\nTraining baseline model...")
    baseline_model = xgb.XGBClassifier(**baseline_params)
    baseline_model.fit(X_train, y_train, verbose=False)
    baseline_pred = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_pred)
    
    # Train optimized model
    print("Training optimized model...")
    optimized_model = xgb.XGBClassifier(**best_params)
    optimized_model.fit(X_train, y_train, verbose=False)
    optimized_pred = optimized_model.predict_proba(X_test)[:, 1]
    optimized_auc = roc_auc_score(y_test, optimized_pred)
    
    # Compare
    print("\nResults on Test Set:")
    print(f"Baseline AUC:   {baseline_auc:.4f}")
    print(f"Optimized AUC:  {optimized_auc:.4f}")
    print(f"Improvement:    {optimized_auc - baseline_auc:.4f} ({(optimized_auc - baseline_auc)/baseline_auc*100:.2f}%)")
    
    if optimized_auc > baseline_auc:
        print("\n✅ Optimized model performs better! Consider using these parameters.")
        return optimized_model, optimized_auc
    else:
        print("\n⚠️ Baseline model still competitive. Difference may not be significant.")
        return baseline_model, baseline_auc


def main():
    """Main hyperparameter tuning pipeline."""
    print("="*60)
    print("HYPERPARAMETER TUNING - XGBOOST NO-SHOW MODEL")
    print("="*60)
    
    # Check Optuna availability
    if not OPTUNA_AVAILABLE:
        print("\n❌ Error: Optuna not installed")
        print("Install with: uv pip install optuna")
        return
    
    # 1. Load and engineer features
    print("\n1. Loading and engineering features...")
    df = load_raw_data('data/raw/noshowappointments-kagglev2-may-2016.csv')
    df_engineered = engineer_features(df)
    
    features = get_feature_columns()
    X = df_engineered[features].copy()
    y = df_engineered['no_show_binary'].copy()
    
    print(f"Dataset shape: {X.shape}")
    
    # 2. Train/test split (80/20)
    print("\n2. Creating train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # 3. Run hyperparameter optimization
    print("\n3. Running hyperparameter optimization...")
    print("This may take 10-30 minutes depending on n_trials...")
    
    best_params, study = tune_hyperparameters(
        X_train, y_train,
        n_trials=100  # Adjust based on time constraints
    )
    
    # 4. Save best parameters
    print("\n4. Saving best parameters...")
    save_best_params(best_params)
    
    # 5. Plot optimization history
    print("\n5. Generating optimization plots...")
    plot_optimization_history(study)
    
    # 6. Compare with baseline
    print("\n6. Comparing with baseline model...")
    best_model, best_auc = compare_with_baseline(
        best_params, X_train, y_train, X_test, y_test
    )
    
    # 7. Save best model
    print("\n7. Saving optimized model...")
    with open('models/xgboost_noshow_model_optimized.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Optimized model saved to models/xgboost_noshow_model_optimized.pkl")
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review best_hyperparameters.json")
    print("2. Update main.py with best parameters if improvement is significant")
    print("3. Retrain full model with optimized parameters")


if __name__ == "__main__":
    main()
