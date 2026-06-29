import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, Rescaling
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
import numpy as np

# Each ImageNet backbone expects its own input preprocessing. The app feeds
# images normalised to [0, 1]; we rescale back to [0, 255] inside the graph and
# then apply the architecture-specific preprocess_input so the pretrained
# weights actually see the distribution they were trained on.
_PREPROCESSORS = {
    'resnet50': resnet_preprocess,
    'densenet121': densenet_preprocess,
    'efficientnetb0': efficientnet_preprocess,
}
_BACKBONES = {
    'resnet50': ResNet50,
    'densenet121': DenseNet121,
    'efficientnetb0': EfficientNetB0,
}

class LungCancerCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, architecture='resnet50'):
        """
        Initialize the CNN model for lung cancer detection
        
        Args:
            input_shape: Input image shape
            num_classes: Number of classes (default: 2 for binary classification)
            architecture: Base architecture ('resnet50', 'densenet121', 'efficientnetb0')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None
        self.base_model = None
        self.history = None
        self.loss = None

    def build_model(self, dropout_rate=0.5, l2_reg=0.001):
        """
        Build the CNN model with transfer learning

        The backbone is built on top of an explicit Input -> Rescaling ->
        preprocess_input chain (via ``input_tensor``) so the convolutional
        layers stay top-level layers of the final model. That keeps Grad-CAM
        working (it can find the last conv layer) and lets fine-tuning unfreeze
        the real backbone.

        Args:
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
        """
        if self.architecture not in _BACKBONES:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        inputs = Input(shape=self.input_shape)
        x = Rescaling(255.0)(inputs)            # [0,1] -> [0,255]
        x = _PREPROCESSORS[self.architecture](x)  # architecture-specific normalisation

        base_model = _BACKBONES[self.architecture](
            weights='imagenet',
            include_top=False,
            input_tensor=x,
        )

        # Freeze base model layers initially
        base_model.trainable = False
        self.base_model = base_model

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

        # Output layer
        if self.num_classes == 2:
            predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        else:
            predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create model
        self.model = Model(inputs=inputs, outputs=predictions)

        return self.model
    
    def compile_model(self, learning_rate=0.001, loss=None, metrics=None):
        """
        Compile the model with appropriate loss function and metrics
        
        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function (auto-selected if None)
            metrics: List of metrics (auto-selected if None)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Auto-select loss function
        if loss is None:
            if self.num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'

        # Auto-select metrics
        if metrics is None:
            metrics = self._default_metrics()

        # Remember loss so fine-tuning can recompile with the same objective
        self.loss = loss

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return self.model

    def _default_metrics(self):
        """Build a fresh list of metrics (fresh objects each call so they can be
        safely reused across a fine-tuning recompile)."""
        if self.num_classes == 2:
            return ['accuracy', AUC(name='auc')]
        return ['accuracy']
    
    def get_callbacks(self, patience=10, monitor='val_loss', save_path='best_model.h5'):
        """
        Get training callbacks
        
        Args:
            patience: Early stopping patience
            monitor: Metric to monitor
            save_path: Path to save best model
        """
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                save_path,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_data, validation_data, epochs=50, class_weight=None, 
              fine_tune_epochs=0, fine_tune_lr=1e-5):
        """
        Train the model with optional fine-tuning
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of epochs for initial training
            class_weight: Class weights for handling imbalance
            fine_tune_epochs: Number of epochs for fine-tuning
            fine_tune_lr: Learning rate for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() and compile_model() first.")
        
        # Initial training with frozen base model
        print("Starting initial training with frozen base model...")
        callbacks = self.get_callbacks()
        
        history1 = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Fine-tuning (if specified)
        if fine_tune_epochs > 0:
            print("\nStarting fine-tuning with unfrozen base model...")

            # Unfreeze the actual pretrained backbone (not just the input layer)
            self.base_model.trainable = True

            # Recompile with a low learning rate, same loss, fresh metric objects
            self.model.compile(
                optimizer=Adam(learning_rate=fine_tune_lr),
                loss=self.loss,
                metrics=self._default_metrics()
            )
            
            # Continue training
            callbacks = self.get_callbacks(patience=5)
            history2 = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
            
            # Combine histories over the union of keys so a key present in one
            # phase but not the other can't KeyError or be silently dropped.
            combined_history = {}
            all_keys = set(history1.history.keys()) | set(history2.history.keys())
            for key in all_keys:
                combined_history[key] = (
                    history1.history.get(key, []) + history2.history.get(key, [])
                )

            self.history = combined_history
        else:
            self.history = history1.history
        
        return self.history
    
    def predict(self, X, batch_size=32):
        """
        Make predictions on input data
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        predictions = self.model.predict(X, batch_size=batch_size)
        
        # Convert predictions based on number of classes
        if self.num_classes == 2:
            # Binary classification
            predicted_probs = predictions.flatten()
            predicted_classes = (predicted_probs > 0.5).astype(int)
        else:
            # Multi-class classification
            predicted_classes = np.argmax(predictions, axis=1)
            predicted_probs = np.max(predictions, axis=1)
        
        return predicted_classes, predicted_probs, predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        return self.model.summary()
