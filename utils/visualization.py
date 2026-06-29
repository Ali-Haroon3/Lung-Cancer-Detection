import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
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
    
    def plot_prediction_confidence_distribution(self, y_pred_proba, y_pred):
        """Plot distribution of prediction confidences.

        Args:
            y_pred_proba: For binary models, a 1D array of P(positive class).
                          For multi-class, a 2D array of per-class probabilities.
            y_pred: 1D array of predicted class indices.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        proba = np.asarray(y_pred_proba)
        preds = np.asarray(y_pred)

        # Confidence = probability assigned to the predicted class
        if proba.ndim == 1:
            confidences = np.where(preds == 1, proba, 1 - proba)
        else:
            confidences = np.max(proba, axis=1)

        # Confidence distribution histogram
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Confidences')
        ax1.grid(True, alpha=0.3)

        # Confidence grouped by predicted class
        class_confidences = {}
        for cls, conf in zip(preds, confidences):
            label = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            class_confidences.setdefault(label, []).append(conf)

        if class_confidences:
            ax2.boxplot(list(class_confidences.values()), labels=list(class_confidences.keys()))
            ax2.set_xlabel('Predicted Class')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Distribution by Class')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    
    @staticmethod
    def _last_conv_layer_name(model):
        """Return the name of the last 4D (conv feature map) layer."""
        for layer in reversed(model.layers):
            try:
                shape = layer.output.shape
            except (AttributeError, ValueError):
                continue
            if len(shape) == 4:
                return layer.name
        return None

    def make_gradcam_heatmap(self, model, image, class_index=None):
        """Compute a real Grad-CAM heatmap for ``image`` (values in [0, 1]).

        Returns a 2D numpy array normalised to [0, 1], or None if the model has
        no convolutional layer to localise on.
        """
        import tensorflow as tf

        last_conv = self._last_conv_layer_name(model)
        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(last_conv).output, model.output]
        )

        x = np.expand_dims(image.astype('float32'), axis=0)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            if preds.shape[-1] == 1:          # binary sigmoid head
                class_channel = preds[:, 0]
            else:                              # multi-class softmax head
                if class_index is None:
                    class_index = int(tf.argmax(preds[0]))
                class_channel = preds[:, class_index]

        grads = tape.gradient(class_channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def create_class_activation_map(self, model, image, class_index=None):
        """Create a Grad-CAM overlay for model interpretability.

        Returns a tuple ``(figure, heatmap)`` so callers can both render the
        plot and inspect the raw heatmap.
        """
        heatmap = self.make_gradcam_heatmap(model, image, class_index)

        # Build a uint8 RGB image for display
        disp = image
        if disp.max() <= 1.0:
            disp = (disp * 255).astype(np.uint8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        if disp.ndim == 3:
            ax1.imshow(disp)
        else:
            ax1.imshow(disp, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')

        if disp.ndim == 3:
            ax2.imshow(disp)
        else:
            ax2.imshow(disp, cmap='gray')

        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (disp.shape[1], disp.shape[0]))
            im = ax2.imshow(heatmap_resized, cmap='jet', alpha=0.5)
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            ax2.set_title('Grad-CAM')
        else:
            ax2.set_title('Grad-CAM unavailable')
        ax2.axis('off')

        plt.tight_layout()
        return fig, heatmap

    def visualize_feature_maps(self, model, image, layer_name=None):
        """Visualize real feature maps from an early convolutional layer."""
        import tensorflow as tf

        conv_layers = []
        for layer in model.layers:
            try:
                if len(layer.output.shape) == 4:
                    conv_layers.append(layer.name)
            except (AttributeError, ValueError):
                continue

        if not conv_layers:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No convolutional layers found',
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig

        # Use an early conv layer (basic edges/textures) by default
        target = layer_name or conv_layers[min(2, len(conv_layers) - 1)]
        feat_model = tf.keras.models.Model(model.inputs, model.get_layer(target).output)

        x = np.expand_dims(image.astype('float32'), axis=0)
        feature_maps = feat_model.predict(x, verbose=0)[0]
        n_maps = min(8, feature_maps.shape[-1])

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        for i in range(8):
            if i < n_maps:
                axes[i].imshow(feature_maps[..., i], cmap='viridis')
                axes[i].set_title(f'Channel {i + 1}', fontsize=9)
            axes[i].axis('off')

        plt.suptitle(f'Feature Maps — layer "{target}"', fontsize=14)
        plt.tight_layout()
        return fig

    def visualize_data_augmentation(self, image, datagen, num_examples=6):
        """Show ``num_examples`` augmented versions of a single image."""
        fig, axes = plt.subplots(1, num_examples, figsize=(3 * num_examples, 3))
        if num_examples == 1:
            axes = [axes]

        x = np.expand_dims(image, axis=0)
        iterator = datagen.flow(x, batch_size=1)
        for i in range(num_examples):
            batch = next(iterator)
            aug = batch[0]
            disp = aug if aug.max() <= 1.0 else aug / 255.0
            disp = np.clip(disp, 0, 1)
            axes[i].imshow(disp)
            axes[i].set_title(f'Augmented {i + 1}', fontsize=9)
            axes[i].axis('off')

        plt.suptitle('Data Augmentation Preview', fontsize=14)
        plt.tight_layout()
        return fig