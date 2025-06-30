import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import cv2
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow safely with delayed loading
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Import database service
from database.db_service import DatabaseService

# Configure page
st.set_page_config(
    page_title="Lung Cancer Detection CNN",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database service
@st.cache_resource
def init_database():
    try:
        return DatabaseService()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

db_service = init_database()

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['normal', 'cancer']
if 'current_model_id' not in st.session_state:
    st.session_state.current_model_id = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Load existing trained model from database if available
if db_service and not st.session_state.model_trained:
    try:
        models_df = db_service.get_models()
        if not models_df.empty:
            trained_models = models_df[models_df['is_trained'] == True]
            if not trained_models.empty:
                latest_model = trained_models.iloc[-1]
                model_path = latest_model.get('model_file_path', 'best_model.h5')
                if os.path.exists(model_path):
                    st.session_state.model_trained = True
                    st.session_state.current_model_id = str(latest_model['id'])
    except Exception:
        pass  # Continue without error if model loading fails
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'data_split' not in st.session_state:
    st.session_state.data_split = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if 'current_model_id' not in st.session_state:
    st.session_state.current_model_id = None
if 'current_dataset_id' not in st.session_state:
    st.session_state.current_dataset_id = None

# Main page
st.title("🫁 Lung Cancer Detection using CNN")
st.markdown("""
### Deep Learning Application for Medical Imaging Analysis

This application uses Convolutional Neural Networks (CNN) to detect lung cancer from medical imaging datasets 
including CT scans and chest X-rays. The system implements transfer learning with pre-trained models and 
provides comprehensive evaluation metrics suitable for medical applications.

#### Features:
- **Data Upload & Preprocessing**: Support for DICOM and standard image formats
- **Model Training**: Transfer learning with ResNet50, DenseNet121, and EfficientNet
- **Evaluation**: Medical-specific metrics including AUC, sensitivity, specificity
- **Prediction**: Real-time inference with confidence scores and interpretability
- **Visualization**: Comprehensive result visualization and model performance analysis

#### Navigation:
Use the sidebar to navigate between different sections of the application.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Database status
st.sidebar.subheader("Database Status")
if db_service:
    st.sidebar.success("✅ Database Connected")
    try:
        stats = db_service.get_dashboard_stats()
        if stats:
            st.sidebar.metric("Total Models", stats.get('total_models', 0))
            st.sidebar.metric("Total Predictions", stats.get('total_predictions', 0))
    except:
        pass
else:
    st.sidebar.error("❌ Database Error")

st.sidebar.markdown("---")

# Model status in sidebar
st.sidebar.subheader("Model Status")
if st.session_state.model_trained:
    st.sidebar.success("✅ Model Trained")
    if st.session_state.model_performance:
        st.sidebar.metric("Test Accuracy", f"{st.session_state.model_performance.get('accuracy', 0):.3f}")
        st.sidebar.metric("Test AUC", f"{st.session_state.model_performance.get('auc', 0):.3f}")
else:
    st.sidebar.warning("⚠️ Model Not Trained")

st.sidebar.markdown("---")

# Quick stats
st.sidebar.subheader("Quick Information")
st.sidebar.info("""
**Supported Formats:**
- DICOM (.dcm)
- PNG, JPG, JPEG
- CT Scans & X-rays

**Model Architectures:**
- ResNet50
- DenseNet121
- EfficientNetB0

**Evaluation Metrics:**
- Accuracy, Precision, Recall
- AUC-ROC, Sensitivity, Specificity
- Confusion Matrix
""")

# Main content area
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("🗂️ Data Management")
    st.write("Upload and preprocess your medical imaging datasets")
    if st.button("Go to Data Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    st.subheader("🧠 Model Training")
    st.write("Train CNN models with transfer learning")
    if st.button("Go to Training", use_container_width=True):
        st.switch_page("pages/2_Model_Training.py")

with col3:
    st.subheader("📊 Evaluation & Prediction")
    st.write("Evaluate model performance and make predictions")
    col3_1, col3_2 = st.columns(2)
    with col3_1:
        if st.button("Evaluation", use_container_width=True):
            st.switch_page("pages/3_Model_Evaluation.py")
    with col3_2:
        if st.button("Prediction", use_container_width=True):
            st.switch_page("pages/4_Prediction.py")

with col4:
    st.subheader("🗄️ Database")
    st.write("View database analytics and manage stored data")
    if st.button("Database Management", use_container_width=True):
        st.switch_page("pages/5_Database_Management.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>Lung Cancer Detection CNN Application | Built with Streamlit & TensorFlow</p>
    <p>⚠️ This application is for research and educational purposes only. Not for clinical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
