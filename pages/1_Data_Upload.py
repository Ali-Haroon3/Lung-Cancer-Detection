import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import zipfile
import tempfile
import shutil
from utils.data_preprocessing import MedicalImagePreprocessor
from utils.visualization import MedicalVisualization
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Data Upload - Lung Cancer Detection",
    page_icon="🗂️",
    layout="wide"
)

st.title("🗂️ Data Upload and Preprocessing")

# Initialize preprocessor
@st.cache_resource
def get_preprocessor():
    return MedicalImagePreprocessor(target_size=(224, 224))

preprocessor = get_preprocessor()

# Initialize session state for data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'data_split' not in st.session_state:
    st.session_state.data_split = None

# Sidebar for navigation
st.sidebar.markdown("### Data Upload Options")
upload_method = st.sidebar.radio(
    "Choose upload method:",
    ["Individual Files", "ZIP Archive", "Sample Dataset"]
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Upload Medical Images")
    
    if upload_method == "Individual Files":
        st.markdown("Upload individual medical images (DICOM, PNG, JPG, etc.)")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose medical image files",
            type=['dcm', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload CT scans, X-rays, or other medical images"
        )
        
        # Class assignment
        if uploaded_files:
            st.markdown("### Class Assignment")
            st.info("Assign each uploaded file to a class (Normal or Cancer)")
            
            file_classes = {}
            for i, file in enumerate(uploaded_files):
                col_file, col_class = st.columns([3, 1])
                with col_file:
                    st.text(file.name)
                with col_class:
                    file_classes[file.name] = st.selectbox(
                        "Class",
                        ["Normal", "Cancer"],
                        key=f"class_{i}"
                    )
            
            if st.button("Process Uploaded Files", type="primary"):
                with st.spinner("Processing uploaded files..."):
                    try:
                        # Create temporary directory
                        temp_dir = tempfile.mkdtemp()
                        
                        # Save files and organize by class
                        for file in uploaded_files:
                            class_name = file_classes[file.name]
                            class_dir = os.path.join(temp_dir, class_name)
                            os.makedirs(class_dir, exist_ok=True)
                            
                            file_path = os.path.join(class_dir, file.name)
                            with open(file_path, 'wb') as f:
                                f.write(file.getbuffer())
                        
                        # Process dataset
                        X, y, class_names = preprocessor.create_dataset_from_directory(temp_dir)
                        
                        # Store in session state
                        st.session_state.uploaded_data = {
                            'X': X,
                            'y': y,
                            'class_names': class_names
                        }
                        st.session_state.class_names = class_names
                        
                        # Clean up
                        shutil.rmtree(temp_dir)
                        
                        st.success(f"Successfully processed {len(X)} images!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
    
    elif upload_method == "ZIP Archive":
        st.markdown("Upload a ZIP file containing organized medical images")
        st.info("""
        **Expected ZIP structure:**
        ```
        dataset.zip
        ├── Normal/
        │   ├── image1.png
        │   ├── image2.dcm
        │   └── ...
        └── Cancer/
            ├── image1.png
            ├── image2.dcm
            └── ...
        ```
        """)
        
        uploaded_zip = st.file_uploader(
            "Choose ZIP file",
            type=['zip'],
            help="ZIP file with organized medical images"
        )
        
        if uploaded_zip and st.button("Extract and Process ZIP", type="primary"):
            with st.spinner("Extracting and processing ZIP file..."):
                try:
                    # Create temporary directory
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "dataset.zip")
                    
                    # Save uploaded ZIP
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_zip.getbuffer())
                    
                    # Extract ZIP
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find the dataset directory
                    extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
                    
                    if len(extracted_dirs) == 1:
                        dataset_dir = os.path.join(temp_dir, extracted_dirs[0])
                    else:
                        dataset_dir = temp_dir
                    
                    # Process dataset
                    X, y, class_names = preprocessor.create_dataset_from_directory(dataset_dir)
                    
                    # Store in session state
                    st.session_state.uploaded_data = {
                        'X': X,
                        'y': y,
                        'class_names': class_names
                    }
                    st.session_state.class_names = class_names
                    
                    # Clean up
                    shutil.rmtree(temp_dir)
                    
                    st.success(f"Successfully processed {len(X)} images from ZIP archive!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing ZIP file: {str(e)}")
    
    else:  # Sample Dataset
        st.markdown("### Load Sample Dataset")
        st.info("Load a pre-generated lung cancer imaging dataset for demonstration")
        
        col_sample1, col_sample2 = st.columns([2, 1])
        
        with col_sample1:
            st.markdown("""
            **Sample Dataset Details:**
            - 100 synthetic lung CT scan images (224x224 pixels)
            - 50 Normal lung images
            - 50 Cancer lung images with nodule indicators
            - Ready for immediate model training
            """)
        
        with col_sample2:
            if st.button("Load Sample Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading sample dataset..."):
                    try:
                        # Load sample dataset from the generated files
                        sample_dir = "sample_data/lung_cancer_dataset"
                        
                        if os.path.exists(sample_dir):
                            # Process sample dataset
                            X, y, class_names = preprocessor.create_dataset_from_directory(sample_dir)
                            
                            # Store in session state
                            st.session_state.uploaded_data = {
                                'X': X,
                                'y': y,
                                'class_names': class_names
                            }
                            st.session_state.class_names = class_names
                            
                            st.success(f"Successfully loaded {len(X)} sample images!")
                            st.rerun()
                        else:
                            st.error("Sample dataset not found. Please generate it first.")
                            
                    except Exception as e:
                        st.error(f"Error loading sample dataset: {str(e)}")

with col2:
    st.markdown("### Upload Guidelines")
    st.info("""
    **Supported Formats:**
    - DICOM (.dcm)
    - PNG, JPG, JPEG
    - BMP, TIFF
    
    **Image Requirements:**
    - Medical images (CT scans, X-rays)
    - Any resolution (will be resized)
    - Grayscale or RGB
    
    **Data Organization:**
    - Organize by classes (Normal/Cancer)
    - Balanced dataset recommended
    - Minimum 20 images per class
    """)
    
    st.markdown("### Preprocessing Steps")
    st.success("""
    ✅ **Automatic Processing:**
    - DICOM handling
    - Image resizing (224x224)
    - Contrast enhancement (CLAHE)
    - Normalization
    - Format standardization
    """)

# Display uploaded data information
if st.session_state.uploaded_data is not None:
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    
    data = st.session_state.uploaded_data
    X, y, class_names = data['X'], data['y'], data['class_names']
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", len(X))
    with col2:
        st.metric("Classes", len(class_names))
    with col3:
        st.metric("Image Size", f"{X.shape[1]}×{X.shape[2]}")
    with col4:
        st.metric("Channels", X.shape[3])
    
    # Class distribution
    st.markdown("### Class Distribution")
    viz = MedicalVisualization(class_names)
    fig_dist = viz.plot_class_distribution(y, "Uploaded Dataset")
    st.pyplot(fig_dist)
    
    # Sample images
    st.markdown("### Sample Images")
    fig_samples = preprocessor.visualize_samples(X, y, class_names, num_samples=8)
    st.pyplot(fig_samples)
    
    # Data splitting
    st.markdown("### 🔄 Data Splitting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.3, 0.2, 0.05)
    with col2:
        val_size = st.slider("Validation Set Size", 0.1, 0.3, 0.2, 0.05)
    with col3:
        random_state = st.number_input("Random State", 1, 1000, 42)
    
    if st.button("Split Dataset", type="primary"):
        with st.spinner("Splitting dataset..."):
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(
                    X, y, test_size=test_size, val_size=val_size, random_state=random_state
                )
                
                # Compute class weights
                class_weights = preprocessor.compute_class_weights(y_train)
                
                # Store split data
                st.session_state.data_split = {
                    'X_train': X_train, 'y_train': y_train,
                    'X_val': X_val, 'y_val': y_val,
                    'X_test': X_test, 'y_test': y_test,
                    'class_weights': class_weights
                }
                
                st.success("Dataset split successfully!")
                
                # Display split statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Set", len(X_train))
                with col2:
                    st.metric("Validation Set", len(X_val))
                with col3:
                    st.metric("Test Set", len(X_test))
                
                # Show class weights
                st.markdown("### Class Weights (for handling imbalance)")
                weights_df = pd.DataFrame([
                    {"Class": class_names[i], "Weight": f"{weight:.3f}"}
                    for i, weight in class_weights.items()
                ])
                st.dataframe(weights_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error splitting dataset: {str(e)}")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.session_state.data_split is not None:
        if st.button("Next: Model Training →", type="primary", use_container_width=True):
            st.switch_page("pages/2_Model_Training.py")
    else:
        st.button("Next: Model Training →", disabled=True, use_container_width=True)
        st.caption("Please upload and split data first")
