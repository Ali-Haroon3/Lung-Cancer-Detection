import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MedicalModelEvaluator:
    def __init__(self, class_names: List[str]):
        """
        Initialize medical model evaluator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: True labels
            
        Returns:
            Dictionary containing evaluation metrics and results
        """
        # Get predictions
        if self.num_classes == 2:
            y_pred_proba = model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # For binary classification
        if self.num_classes == 2:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            specificity = self._calculate_specificity(y_test, y_pred)
            sensitivity = recall  # Same as recall
            
            # AUC-ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # AUC-PR
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
        else:
            # Multi-class metrics
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # For multi-class AUC, we need to binarize
            y_test_bin = label_binarize(y_test, classes=list(range(self.num_classes)))
            if self.num_classes == 2:
                y_test_bin = y_test_bin.ravel()
            
            # Calculate AUC for each class
            roc_auc = {}
            for i in range(self.num_classes):
                if self.num_classes == 2:
                    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba[:, 1])
                    roc_auc[i] = auc(fpr, tpr)
                else:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr, tpr)
            
            # Average AUC
            roc_auc_avg = np.mean(list(roc_auc.values()))
            
            # For simplicity, set sensitivity and specificity as None for multi-class
            sensitivity = None
            specificity = None
            pr_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': roc_auc if self.num_classes == 2 else roc_auc_avg,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return results
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity for binary classification"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def plot_confusion_matrix(self, cm: np.ndarray, normalize=False) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
            cm_display = cm_norm
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
            cm_display = cm
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_display, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Add counts to normalized matrix
        if normalize:
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    ax.text(j+0.5, i+0.7, f'({cm[i,j]})', 
                           ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, results: Dict[str, Any]) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
            roc_auc = results['auc']
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
        else:
            # Multi-class classification
            y_test_bin = label_binarize(results['y_true'], classes=list(range(self.num_classes)))
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], results['y_pred_proba'][:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, 
                       label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, results: Dict[str, Any]) -> plt.Figure:
        """
        Plot Precision-Recall curve (for binary classification)
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Matplotlib figure
        """
        if self.num_classes != 2:
            raise ValueError("Precision-Recall curve is only supported for binary classification")
        
        precision, recall, _ = precision_recall_curve(results['y_true'], results['y_pred_proba'])
        pr_auc = results['pr_auc']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, 
               label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(results['y_true']) / len(results['y_true'])
        ax.axhline(y=baseline, color='red', linestyle='--', 
                  label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, history: Dict[str, List[float]]) -> plt.Figure:
        """
        Plot training history
        
        Args:
            history: Training history dictionary
            
        Returns:
            Matplotlib figure
        """
        metrics = ['loss', 'accuracy']
        if 'auc' in history:
            metrics.append('auc')
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in history and f'val_{metric}' in history:
                epochs = range(1, len(history[metric]) + 1)
                
                axes[i].plot(epochs, history[metric], 'bo-', label=f'Training {metric}')
                axes[i].plot(epochs, history[f'val_{metric}'], 'ro-', label=f'Validation {metric}')
                
                axes[i].set_title(f'Training and Validation {metric.capitalize()}', 
                                fontweight='bold')
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_classification_report_df(self, class_report: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert classification report to DataFrame
        
        Args:
            class_report: Classification report dictionary
            
        Returns:
            Pandas DataFrame
        """
        # Extract per-class metrics
        rows = []
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                rows.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                })
        
        # Add overall metrics
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in class_report:
                metrics = class_report[avg_type]
                rows.append({
                    'Class': avg_type,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def create_medical_metrics_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary of medical-specific metrics
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Pandas DataFrame with medical metrics
        """
        if self.num_classes == 2:
            # Binary classification medical metrics
            metrics_data = {
                'Metric': [
                    'Accuracy', 'Sensitivity (Recall)', 'Specificity', 
                    'Precision (PPV)', 'F1-Score', 'AUC-ROC', 'AUC-PR'
                ],
                'Value': [
                    results['accuracy'],
                    results['sensitivity'],
                    results['specificity'],
                    results['precision'],
                    results['f1_score'],
                    results['auc'],
                    results['pr_auc']
                ],
                'Interpretation': [
                    'Overall correctness',
                    'True positive rate (cancer detection rate)',
                    'True negative rate (healthy correctly identified)',
                    'Positive predictive value',
                    'Harmonic mean of precision and recall',
                    'Area under ROC curve',
                    'Area under Precision-Recall curve'
                ]
            }
        else:
            # Multi-class metrics
            metrics_data = {
                'Metric': [
                    'Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 
                    'F1-Score (Weighted)', 'AUC-ROC (Average)'
                ],
                'Value': [
                    results['accuracy'],
                    results['precision'],
                    results['recall'],
                    results['f1_score'],
                    results['auc']
                ],
                'Interpretation': [
                    'Overall correctness',
                    'Weighted average precision',
                    'Weighted average recall',
                    'Weighted average F1-score',
                    'Average AUC across all classes'
                ]
            }
        
        df = pd.DataFrame(metrics_data)
        df['Value'] = df['Value'].round(4)
        return df
