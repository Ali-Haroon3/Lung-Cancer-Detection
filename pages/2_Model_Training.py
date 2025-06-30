import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from models.cnn_model import LungCancerCNN
from utils.data_preprocessing import MedicalImagePreprocessor
from utils.visualization import MedicalVisualization
from database.db_service import DatabaseService
import matplotlib.pyplot as plt
import time
import os
import json

st.set_page_config(
    page_title="Model Training - Lung Cancer Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CNN Model Training")

# Initialize session state variables
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'current_model_id' not in st.session_state:
    st.session_state.current_model_id = None

# Initialize database service
@st.cache_resource
def get_db_service():
    try:
        return DatabaseService()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

db_service = get_db_service()

# Load existing trained models from database
if db_service and not st.session_state.model_trained:
    try:
        models_df = db_service.get_models()
        if not models_df.empty:
            # Find the most recent trained model
            trained_models = models_df[models_df['is_trained'] == True]
            if not trained_models.empty:
                latest_model = trained_models.iloc[-1]  # Get the most recent
                
                # Check if model file exists
                model_path = latest_model.get('model_file_path', 'best_model.h5')
                if model_path and os.path.exists(model_path):
                    # Load the trained model
                    from models.cnn_model import LungCancerCNN
                    
                    # Restore model configuration
                    input_shape = eval(latest_model['input_shape']) if latest_model['input_shape'] else (224, 224, 3)
                    num_classes = latest_model['num_classes']
                    architecture = latest_model['architecture']
                    
                    # Initialize and load model
                    cnn_model = LungCancerCNN(
                        input_shape=input_shape,
                        num_classes=num_classes, 
                        architecture=architecture
                    )
                    cnn_model.load_model(model_path)
                    
                    # Update session state
                    st.session_state.trained_model = cnn_model
                    st.session_state.model_trained = True
                    st.session_state.current_model_id = str(latest_model['id'])
                    st.session_state.class_names = ['normal', 'cancer']  # Default for lung cancer
                    
                    # Load training history if available
                    training_sessions = db_service.get_training_sessions(str(latest_model['id']))
                    if not training_sessions.empty:
                        latest_session = training_sessions.iloc[-1]
                        if latest_session.get('training_history'):
                            st.session_state.training_history = json.loads(latest_session['training_history'])
    except Exception as e:
        st.info(f"No previous trained model found: {str(e)}")

# Check if data is available
if 'uploaded_data' not in st.session_state or st.session_state.uploaded_data is None:
    st.warning("No dataset found. Let me create sample data for training demonstration.")
    
    # Create sample dataset for demonstration
    if st.button("Create Sample Dataset", type="primary"):
        # Generate synthetic medical imaging data for demonstration
        np.random.seed(42)
        
        # Create sample images (224x224x3) - 200 samples
        n_samples = 200
        img_size = (224, 224, 3)
        
        # Generate sample data
        X_sample = np.random.rand(n_samples, *img_size).astype(np.float32)
        # Add some structure to make it look more like medical images
        for i in range(n_samples):
            # Add circular patterns (simulating lung structures)
            center_x, center_y = np.random.randint(50, 174, 2)
            radius = np.random.randint(20, 60)
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            X_sample[i][mask] *= 1.5
        
        # Create balanced labels (50% normal, 50% cancer)
        y_sample = np.concatenate([
            np.zeros(n_samples//2),  # Normal
            np.ones(n_samples//2)    # Cancer
        ]).astype(int)
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X_sample = X_sample[indices]
        y_sample = y_sample[indices]
        
        # Split the data
        train_split = int(0.7 * n_samples)
        val_split = int(0.85 * n_samples)
        
        X_train = X_sample[:train_split]
        y_train = y_sample[:train_split]
        X_val = X_sample[train_split:val_split]
        y_val = y_sample[train_split:val_split]
        X_test = X_sample[val_split:]
        y_test = y_sample[val_split:]
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_array = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        
        # Store in session state
        st.session_state.data_split = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_weights': class_weights
        }
        st.session_state.class_names = ['Normal', 'Cancer']
        
        # Save to database if available
        if db_service:
            try:
                dataset_id = db_service.save_dataset(
                    name="Sample Training Dataset",
                    X=X_sample,
                    y=y_sample,
                    class_names=['Normal', 'Cancer'],
                    description="Synthetic dataset for model training demonstration"
                )
                if dataset_id:
                    st.session_state.current_dataset_id = dataset_id
                    st.success("Sample dataset created and saved to database!")
                    st.rerun()
            except Exception as e:
                st.warning(f"Dataset created but not saved to database: {str(e)}")
                st.success("Sample dataset created!")
                st.rerun()
        else:
            st.success("Sample dataset created!")
            st.rerun()
    
    st.info("👆 Click the button above to create a sample dataset for training, or go to Data Upload to use your own data.")
    
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Process uploaded data if available
uploaded_data = st.session_state.uploaded_data
class_names = st.session_state.class_names

# Prepare data splits if not already done
if 'data_split' not in st.session_state or st.session_state.data_split is None:
    st.info("Preparing data for training...")
    
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor()
    
    # Split the data
    X, y = uploaded_data['X'], uploaded_data['y']
    
    # Create train/validation/test splits
    data_splits = preprocessor.prepare_data_splits(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )
    
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    # Calculate class weights for imbalanced data
    class_weights = preprocessor.calculate_class_weights(y_train)
    
    # Store in session state
    st.session_state.data_split = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_weights': class_weights
    }
    
    # Save dataset to database
    if db_service:
        try:
            dataset_id = db_service.save_dataset(
                name="Lung Cancer Dataset",
                X=X,
                y=y,
                class_names=class_names,
                description="Lung cancer detection training dataset"
            )
            if dataset_id:
                st.session_state.current_dataset_id = dataset_id
                st.success("Dataset prepared and saved to database!")
        except Exception as e:
            st.warning(f"Dataset prepared but not saved to database: {str(e)}")

# Get prepared data splits
data_split = st.session_state.data_split
X_train = data_split['X_train']
y_train = data_split['y_train']
X_val = data_split['X_val']
y_val = data_split['y_val']
X_test = data_split['X_test']
y_test = data_split['y_test']
class_weights = data_split['class_weights']

# Initialize preprocessor
preprocessor = MedicalImagePreprocessor()

# Sidebar for model configuration
st.sidebar.markdown("### Model Configuration")

architecture = st.sidebar.selectbox(
    "Base Architecture",
    ["resnet50", "densenet121", "efficientnetb0"],
    help="Pre-trained model for transfer learning"
)

# Training parameters
st.sidebar.markdown("### Training Parameters")
epochs = st.sidebar.number_input("Epochs", 1, 100, 20)
batch_size = st.sidebar.number_input("Batch Size", 8, 64, 32)
learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.8, 0.5)
l2_reg = st.sidebar.number_input("L2 Regularization", 0.0001, 0.01, 0.001, format="%.4f")

# Data augmentation
st.sidebar.markdown("### Data Augmentation")
use_augmentation = st.sidebar.checkbox("Enable Data Augmentation", True)

# Advanced training options
st.sidebar.markdown("### Advanced Options")
use_class_weights = st.sidebar.checkbox("Use Class Weights", True)
fine_tune = st.sidebar.checkbox("Fine-tuning", False)
if fine_tune:
    fine_tune_epochs = st.sidebar.number_input("Fine-tune Epochs", 1, 50, 10)
    fine_tune_lr = st.sidebar.number_input("Fine-tune LR", 0.00001, 0.001, 0.0001, format="%.5f")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Model Architecture Overview")
    
    # Display model info
    model_info = {
        "Base Architecture": architecture.upper(),
        "Input Shape": f"{X_train.shape[1:]}",
        "Number of Classes": len(class_names),
        "Total Training Samples": len(X_train),
        "Total Validation Samples": len(X_val),
        "Batch Size": batch_size,
        "Expected Steps per Epoch": len(X_train) // batch_size
    }
    
    info_df = pd.DataFrame([
        {"Parameter": k, "Value": str(v)} for k, v in model_info.items()
    ])
    st.dataframe(info_df, use_container_width=True)

with col2:
    st.markdown("### Training Status")
    
    if st.session_state.model_trained:
        st.success("✅ Model Trained")
        if st.session_state.training_history:
            last_epoch = len(st.session_state.training_history['loss'])
            st.metric("Epochs Completed", last_epoch)
            if 'val_accuracy' in st.session_state.training_history:
                final_val_acc = st.session_state.training_history['val_accuracy'][-1]
                st.metric("Final Val Accuracy", f"{final_val_acc:.3f}")
    else:
        st.warning("⚠️ Model Not Trained")
    
    # Model parameters
    st.markdown("### Current Settings")
    st.text(f"Architecture: {architecture}")
    st.text(f"Learning Rate: {learning_rate}")
    st.text(f"Dropout: {dropout_rate}")
    st.text(f"Batch Size: {batch_size}")

# Training section
st.markdown("---")
st.markdown("### 🚀 Start Training")

if st.button("Start Training", type="primary", use_container_width=True):
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    
    try:
        status_text.text("Initializing model...")
        
        # Initialize model
        model = LungCancerCNN(
            input_shape=X_train.shape[1:],
            num_classes=len(class_names),
            architecture=architecture
        )
        
        # Build and compile model
        model.build_model(dropout_rate=dropout_rate, l2_reg=l2_reg)
        model.compile_model(learning_rate=learning_rate)
        
        status_text.text("Creating data generators...")
        
        # Create data generators
        train_gen, val_gen = preprocessor.create_data_generators(
            X_train, y_train, X_val, y_val,
            batch_size=batch_size, augment=use_augmentation
        )
        
        status_text.text("Starting training...")
        
        # Prepare class weights
        class_weight_dict = class_weights if use_class_weights else None
        
        # Custom callback to update progress
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs, progress_bar, status_text, metrics_container):
                self.total_epochs = total_epochs
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.metrics_container = metrics_container
                
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.total_epochs
                self.progress_bar.progress(progress)
                
                self.status_text.text(f"Epoch {epoch + 1}/{self.total_epochs}")
                
                # Display current metrics
                if logs:
                    metrics_data = []
                    for key, value in logs.items():
                        metrics_data.append({
                            "Metric": key,
                            "Value": f"{value:.4f}"
                        })
                    metrics_df = pd.DataFrame(metrics_data)
                    self.metrics_container.dataframe(metrics_df, use_container_width=True)
        
        # Add custom callback
        callbacks = model.get_callbacks()
        callbacks.append(StreamlitCallback(epochs, progress_bar, status_text, metrics_container))
        
        # Train model
        if fine_tune:
            history = model.train(
                train_gen, val_gen,
                epochs=epochs,
                class_weight=class_weight_dict,
                fine_tune_epochs=fine_tune_epochs if fine_tune else 0,
                fine_tune_lr=fine_tune_lr if fine_tune else 0.0001
            )
        else:
            history = model.train(
                train_gen, val_gen,
                epochs=epochs,
                class_weight=class_weight_dict
            )
        
        # Store results in session state
        st.session_state.trained_model = model
        st.session_state.model_trained = True
        st.session_state.training_history = history
        
        # Save to database
        if db_service:
            try:
                # Save model metadata
                model_name = f"{architecture.upper()}_LungCancer_{int(time.time())}"
                model_id = db_service.save_model(
                    name=model_name,
                    architecture=architecture,
                    input_shape=X_train.shape[1:],
                    num_classes=len(class_names)
                )
                
                if model_id:
                    st.session_state.current_model_id = model_id
                    
                    # Update model status
                    db_service.update_model_status(model_id, is_trained=True)
                    
                    # Save training session
                    training_params = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'dropout_rate': dropout_rate,
                        'l2_reg': l2_reg,
                        'use_augmentation': use_augmentation,
                        'use_class_weights': use_class_weights,
                        'class_weights': class_weights,
                        'fine_tune': fine_tune,
                        'fine_tune_epochs': fine_tune_epochs if fine_tune else None,
                        'fine_tune_lr': fine_tune_lr if fine_tune else None
                    }
                    
                    # Convert numpy/tensorflow types to Python types for JSON serialization
                    training_results = {
                        'final_train_loss': float(history['loss'][-1]),
                        'final_train_accuracy': float(history['accuracy'][-1]),
                        'final_val_loss': float(history['val_loss'][-1]),
                        'final_val_accuracy': float(history['val_accuracy'][-1]),
                        'history': {k: [float(v) for v in values] for k, values in history.items()}
                    }
                    
                    session_id = db_service.save_training_session(
                        model_id=model_id,
                        dataset_id=st.session_state.get('current_dataset_id', 'unknown'),
                        params=training_params,
                        results=training_results
                    )
                    
                    if session_id:
                        status_text.text("Training completed and saved to database!")
                    else:
                        status_text.text("Training completed but not saved to database!")
                else:
                    status_text.text("Training completed but model not saved to database!")
                    
            except Exception as e:
                status_text.text(f"Training completed but database save failed: {str(e)}")
        else:
            status_text.text("Training completed!")
        
        # Update progress
        progress_bar.progress(1.0)
        
        st.success("Model training completed successfully!")
        
        # Display training summary
        st.markdown("### Training Summary")
        final_metrics = {
            "Final Training Loss": f"{history['loss'][-1]:.4f}",
            "Final Validation Loss": f"{history['val_loss'][-1]:.4f}",
            "Final Training Accuracy": f"{history['accuracy'][-1]:.4f}",
            "Final Validation Accuracy": f"{history['val_accuracy'][-1]:.4f}",
            "Total Epochs": len(history['loss']),
            "Best Validation Loss": f"{min(history['val_loss']):.4f}"
        }
        
        summary_df = pd.DataFrame([
            {"Metric": k, "Value": str(v)} for k, v in final_metrics.items()
        ])
        st.dataframe(summary_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        metrics_container.empty()

# Display training history if available
if st.session_state.training_history is not None:
    st.markdown("---")
    st.markdown("### 📈 Training History")
    
    history = st.session_state.training_history
    viz = MedicalVisualization(class_names)
    
    # Plot training history
    fig_history = viz.plot_training_history(history)
    st.pyplot(fig_history)
    
    # Training metrics table
    st.markdown("### Training Metrics by Epoch")
    
    # Create DataFrame from history
    epochs_data = []
    for i in range(len(history['loss'])):
        epoch_data = {"Epoch": i + 1}
        for key, values in history.items():
            if i < len(values):
                epoch_data[key] = f"{values[i]:.4f}"
        epochs_data.append(epoch_data)
    
    epochs_df = pd.DataFrame(epochs_data)
    st.dataframe(epochs_df, use_container_width=True)

# Model architecture display
if st.session_state.trained_model is not None:
    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    
    if st.button("Show Model Summary"):
        model_summary = st.session_state.trained_model.get_model_summary()
        # Capture the summary
        stringlist = []
        st.session_state.trained_model.model.summary(print_fn=lambda x: stringlist.append(x))
        summary_string = '\n'.join(stringlist)
        
        st.text(summary_string)
    
    # Model saving
    st.markdown("### 💾 Save Model")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Model", use_container_width=True):
            try:
                model_path = "trained_lung_cancer_model.h5"
                st.session_state.trained_model.save_model(model_path)
                st.success(f"Model saved as {model_path}")
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
    
    with col2:
        # Download model (placeholder - in real deployment, would provide download link)
        st.button("Download Model", disabled=True, use_container_width=True)
        st.caption("Download functionality would be implemented in production")

# Data augmentation visualization
if use_augmentation and len(X_train) > 0:
    st.markdown("---")
    st.markdown("### 🔄 Data Augmentation Preview")
    
    # Show augmentation examples
    if st.button("Show Augmentation Examples"):
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            # Create augmentation generator
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2]
            )
        except ImportError:
            st.error("TensorFlow import error. Please restart the application.")
            st.stop()
        
        # Select a random image
        sample_idx = np.random.randint(0, len(X_train))
        sample_image = X_train[sample_idx]
        
        viz = MedicalVisualization(class_names)
        fig_aug = viz.visualize_data_augmentation(sample_image, datagen, num_examples=6)
        st.pyplot(fig_aug)

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to Data Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")
with col2:
    if st.session_state.model_trained:
        if st.button("Next: Model Evaluation →", type="primary", use_container_width=True):
            st.switch_page("pages/3_Model_Evaluation.py")
    else:
        st.button("Next: Model Evaluation →", disabled=True, use_container_width=True)
        st.caption("Please train model first")
