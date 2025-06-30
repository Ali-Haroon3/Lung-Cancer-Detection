import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MedicalVisualization:
    def __init__(self, class_names):
        self.class_names = class_names
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_history(self, history):
        """Plot training history with loss and accuracy"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Training and validation loss
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Training and validation accuracy
        ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate (if available)
        if 'lr' in history:
            ax3.plot(epochs, history['lr'], 'g-', label='Learning Rate')
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.legend()
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Learning Rate')
        
        # Loss difference
        if 'val_loss' in history:
            loss_diff = [val - train for train, val in zip(history['loss'], history['val_loss'])]
            ax4.plot(epochs, loss_diff, 'purple', label='Val Loss - Train Loss')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax4.set_title('Overfitting Monitor')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Difference')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Validation Loss\nNot Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Overfitting Monitor')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        return fig
    
    def plot_roc_curve(self, fpr, tpr, auc_score):
        """Plot ROC curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return fig
    
    def plot_class_distribution(self, y, title="Class Distribution"):
        """Plot class distribution"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        unique, counts = np.unique(y, return_counts=True)
        class_labels = [self.class_names[i] for i in unique]
        
        bars = ax.bar(class_labels, counts, color=['skyblue', 'lightcoral'])
        ax.set_title(title)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
        
        return fig
    
    def plot_sample_images(self, X, y, n_samples=12, title="Sample Images"):
        """Plot sample images from dataset"""
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()
        
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image = X[idx]
            label = self.class_names[y[idx]]
            
            # Handle different image formats
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                axes[i].imshow(image)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Grayscale image
                axes[i].imshow(image.squeeze(), cmap='gray')
            else:
                # 2D grayscale image
                axes[i].imshow(image, cmap='gray')
            
            axes[i].set_title(f'{label}')
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, metrics_dict, title="Model Performance Metrics"):
        """Plot comparison of different metrics"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        bars = ax.bar(metrics, values, color='lightblue', edgecolor='navy')
        ax.set_title(title)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def visualize_predictions(self, images, true_labels, pred_labels, pred_probs, num_samples=8):
        """Visualize predictions with images, true labels, and predictions"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Display image
            img = images[i]
            if len(img.shape) == 3:
                axes[i].imshow(img)
            else:
                axes[i].imshow(img, cmap='gray')
            
            # Get labels and probabilities
            true_label = self.class_names[true_labels[i]] if true_labels[i] < len(self.class_names) else 'Unknown'
            pred_label = self.class_names[pred_labels[i]] if pred_labels[i] < len(self.class_names) else 'Unknown'
            
            # Handle different probability formats
            if len(pred_probs.shape) > 1 and pred_probs.shape[1] > 1:
                confidence = np.max(pred_probs[i])
            else:
                confidence = pred_probs[i] if len(pred_probs.shape) == 1 else pred_probs[i][0]
            
            # Set title with color coding
            color = 'green' if true_label == pred_label else 'red'
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}'
            
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Prediction Results', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_prediction_confidence_distribution(self, predictions, class_names):
        """Plot distribution of prediction confidences"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract confidence scores and predictions
        confidences = [pred['confidence_score'] for pred in predictions]
        predicted_classes = [pred['predicted_class'] for pred in predictions]
        
        # Confidence distribution histogram
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Confidences')
        ax1.grid(True, alpha=0.3)
        
        # Confidence by class
        class_confidences = {}
        for pred_class, conf in zip(predicted_classes, confidences):
            if pred_class not in class_confidences:
                class_confidences[pred_class] = []
            class_confidences[pred_class].append(conf)
        
        if class_confidences:
            ax2.boxplot(list(class_confidences.values()), labels=list(class_confidences.keys()))
            ax2.set_xlabel('Predicted Class')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Distribution by Class')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_class_activation_map(self, model, image, class_index=None):
        """Create class activation map for model interpretability"""
        # Simplified CAM visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Original image
        if len(image.shape) == 3:
            ax1.imshow(image)
        else:
            ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Mock heatmap for demonstration (in real implementation would use gradCAM)
        heatmap = np.random.random((image.shape[0], image.shape[1]))
        im = ax2.imshow(heatmap, cmap='jet', alpha=0.6)
        if len(image.shape) == 3:
            ax2.imshow(image, alpha=0.4)
        else:
            ax2.imshow(image, cmap='gray', alpha=0.4)
        ax2.set_title('Activation Map')
        ax2.axis('off')
        
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig
    
    def visualize_feature_maps(self, model, image, layer_name=None):
        """Visualize feature maps from intermediate layers"""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        # Mock feature maps for demonstration
        for i in range(8):
            feature_map = np.random.random((32, 32))
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Feature Map {i+1}')
            axes[i].axis('off')
        
        plt.suptitle('Feature Maps Visualization', fontsize=14)
        plt.tight_layout()
        return fig