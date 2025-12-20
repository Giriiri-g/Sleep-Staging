"""
Train Traditional ML Classifiers
=================================
Train SVM, Random Forest, and XGBoost classifiers for comparison with neural network.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from tqdm import tqdm

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from extract_sequence_features import prepare_ml_features
from train_comorbidity_classifier import (
    plot_confusion_matrices, plot_roc_curves, plot_pr_curves,
    plot_class_performance, create_stratified_splits
)


class MultiOutputClassifier:
    """Wrapper for sklearn classifiers to handle multi-output binary classification"""
    
    def __init__(self, base_classifier, **kwargs):
        self.base_classifier = base_classifier
        self.kwargs = kwargs
        self.classifiers = []
        self.scalers = []
    
    def fit(self, X, y):
        """Fit classifiers for each output"""
        n_classes = y.shape[1]
        self.classifiers = []
        self.scalers = []
        
        for i in range(n_classes):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers.append(scaler)
            
            clf = self.base_classifier(**self.kwargs)
            clf.fit(X_scaled, y[:, i])
            self.classifiers.append(clf)
        
        return self
    
    def predict(self, X):
        """Predict for all outputs"""
        predictions = []
        for i, (clf, scaler) in enumerate(zip(self.classifiers, self.scalers)):
            X_scaled = scaler.transform(X)
            pred = clf.predict(X_scaled)
            predictions.append(pred)
        return np.column_stack(predictions)
    
    def predict_proba(self, X):
        """Predict probabilities for all outputs"""
        probabilities = []
        for i, (clf, scaler) in enumerate(zip(self.classifiers, self.scalers)):
            X_scaled = scaler.transform(X)
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_scaled)[:, 1]  # Probability of positive class
            else:
                # For SVM, use decision function
                decision = clf.decision_function(X_scaled)
                proba = 1 / (1 + np.exp(-decision))  # Sigmoid transform
            probabilities.append(proba)
        return np.column_stack(probabilities)


def train_svm(X_train, y_train, X_val, y_val, class_weight='balanced'):
    """Train SVM classifier"""
    print("\nTraining SVM...")
    svm = MultiOutputClassifier(
        SVC,
        kernel='rbf',
        probability=True,
        class_weight=class_weight,
        random_state=42
    )
    svm.fit(X_train, y_train)
    
    # Evaluate
    y_pred_val = svm.predict(X_val)
    y_proba_val = svm.predict_proba(X_val)
    
    return svm, y_pred_val, y_proba_val


def train_random_forest(X_train, y_train, X_val, y_val, class_weight='balanced', n_estimators=200):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest...")
    rf = MultiOutputClassifier(
        RandomForestClassifier,
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_val = rf.predict(X_val)
    y_proba_val = rf.predict_proba(X_val)
    
    return rf, y_pred_val, y_proba_val


def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight=None):
    """Train XGBoost classifier"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    
    print("\nTraining XGBoost...")
    n_classes = y_train.shape[1]
    classifiers = []
    scalers = []
    
    for i in range(n_classes):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        scalers.append(scaler)
        
        # Compute scale_pos_weight
        if scale_pos_weight is None:
            pos_count = y_train[:, i].sum()
            neg_count = len(y_train) - pos_count
            if pos_count > 0:
                sw = neg_count / pos_count
            else:
                sw = 1.0
        else:
            sw = scale_pos_weight[i] if isinstance(scale_pos_weight, (list, np.ndarray)) else scale_pos_weight
        
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=sw,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        clf.fit(
            X_train_scaled, y_train[:, i],
            eval_set=[(X_val_scaled, y_val[:, i])],
            verbose=False
        )
        classifiers.append(clf)
    
    xgb_model = type('XGBModel', (), {
        'classifiers': classifiers,
        'scalers': scalers,
        'predict': lambda self, X: np.column_stack([
            clf.predict(scaler.transform(X)) 
            for clf, scaler in zip(self.classifiers, self.scalers)
        ]),
        'predict_proba': lambda self, X: np.column_stack([
            clf.predict_proba(scaler.transform(X))[:, 1]
            for clf, scaler in zip(self.classifiers, self.scalers)
        ])
    })()
    
    # Evaluate
    y_pred_val = xgb_model.predict(X_val)
    y_proba_val = xgb_model.predict_proba(X_val)
    
    return xgb_model, y_pred_val, y_proba_val


def evaluate_model(y_true, y_pred, y_proba, target_names):
    """Evaluate model and return metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=None, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=None, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=None, zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'roc_auc': [],
        'predictions': y_pred,
        'probabilities': y_proba,
        'targets': y_true
    }
    
    # ROC AUC per class
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_proba[:, i])
            metrics['roc_auc'].append(auc)
        except ValueError:
            metrics['roc_auc'].append(0.0)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Traditional ML Classifiers')
    parser.add_argument('--csv_path', type=str, default='mesa_final.csv',
                        help='Path to CSV file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/ml_classifiers',
                        help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_stratified_split', action='store_true',
                        help='Use stratified sampling')
    parser.add_argument('--classifiers', type=str, nargs='+',
                        default=['svm', 'rf', 'xgb'],
                        choices=['svm', 'rf', 'xgb'],
                        help='Classifiers to train')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y, feature_names = prepare_ml_features(args.csv_path)
    
    target_cols = ['insomnia', 'restless leg', 'apnea']
    
    # Split data
    if args.use_stratified_split:
        print("Using stratified sampling...")
        df = pd.read_csv(args.csv_path)
        train_df, val_df, test_df = create_stratified_splits(
            df, target_cols,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.seed
        )
        
        # Get indices
        train_indices = train_df.index.values
        val_indices = val_df.index.values
        test_indices = test_df.index.values
        
        X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    else:
        # Simple split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=args.val_size / (1 - args.test_size),
            random_state=args.seed
        )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Print class distribution
    print("\nClass distribution:")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        print(f"\n{split_name}:")
        for i, col in enumerate(target_cols):
            pos_count = y_split[:, i].sum()
            pos_pct = 100 * pos_count / len(y_split)
            print(f"  {col}: {pos_count} positive ({pos_pct:.1f}%)")
    
    # Train classifiers
    results = {}
    
    if 'svm' in args.classifiers:
        svm_model, svm_pred_val, svm_proba_val = train_svm(X_train, y_train, X_val, y_val)
        svm_metrics_val = evaluate_model(y_val, svm_pred_val, svm_proba_val, target_cols)
        
        # Test evaluation
        svm_pred_test = svm_model.predict(X_test)
        svm_proba_test = svm_model.predict_proba(X_test)
        svm_metrics_test = evaluate_model(y_test, svm_pred_test, svm_proba_test, target_cols)
        
        results['svm'] = {
            'model': svm_model,
            'val_metrics': svm_metrics_val,
            'test_metrics': svm_metrics_test
        }
        
        print("\nSVM Validation Results:")
        print(f"  Macro F1: {svm_metrics_val['macro_f1']:.4f}")
        print(f"  Per-class F1: {svm_metrics_val['f1']}")
        print(f"  ROC AUC: {svm_metrics_val['roc_auc']}")
        
        print("\nSVM Test Results:")
        print(f"  Macro F1: {svm_metrics_test['macro_f1']:.4f}")
        print(f"  Per-class F1: {svm_metrics_test['f1']}")
        print(f"  ROC AUC: {svm_metrics_test['roc_auc']}")
    
    if 'rf' in args.classifiers:
        rf_model, rf_pred_val, rf_proba_val = train_random_forest(X_train, y_train, X_val, y_val)
        rf_metrics_val = evaluate_model(y_val, rf_pred_val, rf_proba_val, target_cols)
        
        # Test evaluation
        rf_pred_test = rf_model.predict(X_test)
        rf_proba_test = rf_model.predict_proba(X_test)
        rf_metrics_test = evaluate_model(y_test, rf_pred_test, rf_proba_test, target_cols)
        
        results['rf'] = {
            'model': rf_model,
            'val_metrics': rf_metrics_val,
            'test_metrics': rf_metrics_test
        }
        
        print("\nRandom Forest Validation Results:")
        print(f"  Macro F1: {rf_metrics_val['macro_f1']:.4f}")
        print(f"  Per-class F1: {rf_metrics_val['f1']}")
        print(f"  ROC AUC: {rf_metrics_val['roc_auc']}")
        
        print("\nRandom Forest Test Results:")
        print(f"  Macro F1: {rf_metrics_test['macro_f1']:.4f}")
        print(f"  Per-class F1: {rf_metrics_test['f1']}")
        print(f"  ROC AUC: {rf_metrics_test['roc_auc']}")
    
    if 'xgb' in args.classifiers:
        if not XGBOOST_AVAILABLE:
            print("\nSkipping XGBoost (not available)")
        else:
            # Compute scale_pos_weight
            scale_pos_weight = []
            for i in range(y_train.shape[1]):
                pos_count = y_train[:, i].sum()
                neg_count = len(y_train) - pos_count
                if pos_count > 0:
                    scale_pos_weight.append(neg_count / pos_count)
                else:
                    scale_pos_weight.append(1.0)
            
            xgb_model, xgb_pred_val, xgb_proba_val = train_xgboost(
                X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos_weight
            )
            xgb_metrics_val = evaluate_model(y_val, xgb_pred_val, xgb_proba_val, target_cols)
            
            # Test evaluation
            xgb_pred_test = xgb_model.predict(X_test)
            xgb_proba_test = xgb_model.predict_proba(X_test)
            xgb_metrics_test = evaluate_model(y_test, xgb_pred_test, xgb_proba_test, target_cols)
            
            results['xgb'] = {
                'model': xgb_model,
                'val_metrics': xgb_metrics_val,
                'test_metrics': xgb_metrics_test
            }
            
            print("\nXGBoost Validation Results:")
            print(f"  Macro F1: {xgb_metrics_val['macro_f1']:.4f}")
            print(f"  Per-class F1: {xgb_metrics_val['f1']}")
            print(f"  ROC AUC: {xgb_metrics_val['roc_auc']}")
            
            print("\nXGBoost Test Results:")
            print(f"  Macro F1: {xgb_metrics_test['macro_f1']:.4f}")
            print(f"  Per-class F1: {xgb_metrics_test['f1']}")
            print(f"  ROC AUC: {xgb_metrics_test['roc_auc']}")
    
    # Generate visualizations for each classifier
    print("\nGenerating visualizations...")
    for clf_name, result in results.items():
        clf_dir = os.path.join(args.output_dir, clf_name)
        os.makedirs(clf_dir, exist_ok=True)
        
        # Test set visualizations
        test_metrics = result['test_metrics']
        plot_confusion_matrices(test_metrics, target_cols, clf_dir)
        plot_roc_curves(test_metrics, target_cols, clf_dir)
        plot_pr_curves(test_metrics, target_cols, clf_dir)
        plot_class_performance(test_metrics, target_cols, clf_dir)
        
        # Save results
        results_dict = {
            'classifier': clf_name,
            'test_metrics': {
                'accuracy': float(test_metrics['accuracy']),
                'macro_precision': float(test_metrics['macro_precision']),
                'macro_recall': float(test_metrics['macro_recall']),
                'macro_f1': float(test_metrics['macro_f1']),
                'per_class': {
                    name: {
                        'precision': float(test_metrics['precision'][i]),
                        'recall': float(test_metrics['recall'][i]),
                        'f1': float(test_metrics['f1'][i]),
                        'roc_auc': float(test_metrics['roc_auc'][i])
                    }
                    for i, name in enumerate(target_cols)
                }
            },
            'val_metrics': {
                'macro_f1': float(result['val_metrics']['macro_f1']),
                'per_class': {
                    name: {
                        'f1': float(result['val_metrics']['f1'][i]),
                        'roc_auc': float(result['val_metrics']['roc_auc'][i])
                    }
                    for i, name in enumerate(target_cols)
                }
            }
        }
        
        with open(os.path.join(clf_dir, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    # Create comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for clf_name, result in results.items():
        test_metrics = result['test_metrics']
        print(f"\n{clf_name.upper()}:")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Macro Recall: {test_metrics['macro_recall']:.4f}")
        print(f"  Mean ROC AUC: {np.mean(test_metrics['roc_auc']):.4f}")
        print(f"  Per-class F1: {test_metrics['f1']}")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()

