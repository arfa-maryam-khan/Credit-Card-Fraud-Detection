"""
Credit Card Fraud Detection - Training Pipeline
MLOps Project with W&B Integration
Fixed version with optimal threshold tuning
"""

import os
import argparse
import pandas as pd
import numpy as np
import wandb
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    auc
)
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def download_from_kaggle():
    """Download dataset from Kaggle using kaggle API"""
    print("Downloading dataset from Kaggle...")
    
    try:
        import kaggle
        
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path='data/',
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully from Kaggle!")
        return 'data/creditcard.csv'
    
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {str(e)}")
        print("\nTo use Kaggle API:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Get API credentials from: https://www.kaggle.com/settings/account")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        raise


def load_data(data_path):
    """Load and perform basic data validation"""
    print("Loading data...")
    
    if not os.path.exists(data_path):
        print(f"⚠️  Dataset not found at {data_path}")
        print("Attempting to download from Kaggle...")
        data_path = download_from_kaggle()
    
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")
    print(f"Legitimate cases: {(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.2f}%)")
    
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Split and scale the data"""
    print("\nPreparing data...")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, config):
    """Train LightGBM model"""
    print("\nTraining model...")
    
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight='balanced',
        random_state=config['random_state'],
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    print("Training completed!")
    
    return model


def find_optimal_threshold(y_test, y_pred_proba, target_recall=0.80):
    """Find threshold that maximizes F1 while maintaining target recall"""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find thresholds where recall >= target
    valid_indices = recalls >= target_recall
    
    if not np.any(valid_indices):
        print(f"Warning: Cannot achieve {target_recall*100}% recall")
        best_idx = np.argmax(f1_scores[:-1])
        return thresholds[best_idx]
    
    # Among valid thresholds, find the one with best F1
    valid_f1 = f1_scores[:-1][valid_indices[:-1]]
    valid_thresholds = thresholds[valid_indices[:-1]]
    
    best_idx = np.argmax(valid_f1)
    optimal_threshold = valid_thresholds[best_idx]
    
    print(f"\n🎯 Optimal threshold: {optimal_threshold:.4f}")
    print(f"Expected Precision: {precisions[:-1][valid_indices[:-1]][best_idx]:.4f}")
    print(f"Expected Recall: {recalls[:-1][valid_indices[:-1]][best_idx]:.4f}")
    print(f"Expected F1: {valid_f1[best_idx]:.4f}")
    
    return optimal_threshold


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    print("\nEvaluating model...")
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, target_recall=0.80)
    
    # Predict with optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold
    }
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE (Optimized)")
    print("="*50)
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*50)
    
    return metrics, y_pred_proba


def plot_confusion_matrix(cm):
    """Create confusion matrix plot"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return 'confusion_matrix.png'


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig('feature_importance.png')
    plt.close()
    
    return 'feature_importance.png'


def save_model(model, scaler, threshold, path='models/artifacts'):
    """Save model, scaler, and threshold"""
    os.makedirs(path, exist_ok=True)
    
    model_path = os.path.join(path, 'model.pkl')
    scaler_path = os.path.join(path, 'scaler.pkl')
    threshold_path = os.path.join(path, 'threshold.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(threshold_path, 'wb') as f:
        pickle.dump(threshold, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Threshold saved to: {threshold_path}")
    
    return model_path, scaler_path


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data_path', type=str, default='data/creditcard.csv',
                        help='Path to the dataset')
    parser.add_argument('--project_name', type=str, default='fraud-detection-mlops',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'class_weight': 'balanced',
        'random_state': 42,
        'test_size': 0.2
    }
    
    # Initialize W&B
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.project_name,
        name=run_name,
        config=config,
        tags=['lightgbm', 'fraud-detection', 'optimized-threshold']
    )
    
    # Load data
    df = load_data(args.data_path)
    
    # Log dataset info to W&B
    wandb.log({
        'dataset_size': len(df),
        'fraud_percentage': df['Class'].sum() / len(df) * 100,
        'n_features': len(df.columns) - 1
    })
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, 
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Train model
    model = train_model(X_train, y_train, config)
    
    # Evaluate model
    metrics, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Log metrics to W&B
    wandb.log({
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'optimal_threshold': metrics['optimal_threshold']
    })
    
    # Create and log confusion matrix
    cm_plot = plot_confusion_matrix(metrics['confusion_matrix'])
    wandb.log({'confusion_matrix': wandb.Image(cm_plot)})
    
    # Create and log feature importance
    feature_names = df.drop('Class', axis=1).columns.tolist()
    fi_plot = plot_feature_importance(model, feature_names)
    wandb.log({'feature_importance': wandb.Image(fi_plot)})
    
    # Save model locally
    model_path, scaler_path = save_model(model, scaler, metrics['optimal_threshold'])
    
    # Log model to W&B
    artifact = wandb.Artifact(
        name='fraud-detection-model',
        type='model',
        description='LightGBM model for credit card fraud detection',
        metadata={
            'framework': 'lightgbm',
            'optimal_threshold': float(metrics['optimal_threshold']),
            'metrics': {
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'roc_auc': float(metrics['roc_auc'])
            }
        }
    )
    
    artifact.add_file(model_path)
    artifact.add_file(scaler_path)
    wandb.log_artifact(artifact)
    
    # Link to model registry (if metrics are good)
    if metrics['f1_score'] > 0.70 and metrics['recall'] > 0.70:
        print("\n✅ Model meets performance criteria!")
        print("Linking to model registry...")
        wandb.log_artifact(artifact, aliases=['latest', 'production'])
    else:
        print("\n⚠️  Model does not meet performance criteria.")
        print("Not promoting to production.")
    
    print("\n🎉 Training pipeline completed!")
    print(f"View results at: {wandb.run.get_url()}")
    
    wandb.finish()


if __name__ == "__main__":
    main()