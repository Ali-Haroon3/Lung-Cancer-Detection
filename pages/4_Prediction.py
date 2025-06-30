import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tempfile
import os
from utils.data_preprocessing import MedicalImagePreprocessor
from utils.visualization import MedicalVisualization
from utils.evaluation import MedicalModelEvaluator
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(
    page_title="Prediction - Lung Cancer Detection",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Lung Cancer Prediction")

# Initialize session state variables
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['normal', 'cancer']

# Check if model is trained
if not st.session_state.model_trained or st.session_state.trained_model is None:
    st.error("No trained model found. Please train a model first.")
    if st.button("Go to Model Training"):
        st.switch_page("pages/2_Model_Training.py")
    st.stop()

# Get model and class names from session state
model = st.session_state.trained_model
class_names = st.session_state.class_names

# Initialize preprocessor and visualizer
preprocessor = MedicalImagePreprocessor(target_size=(224, 224))
viz = MedicalVisualization(class_names)

# Initialize session state for predictions
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []

# Sidebar for prediction options
st.sidebar.markdown("### Prediction Options")

prediction_mode = st.sidebar.radio(
    "Prediction Mode:",
    ["Single Image", "Batch Images"],
    help="Choose between single image or batch prediction"
)

# Confidence threshold for binary classification
if len(class_names) == 2:
    confidence_threshold = st.sidebar.slider(
        "Classification Threshold", 
        0.0, 1.0, 0.5, 0.05,
        help="Threshold for binary classification (Cancer vs Normal)"
    )
else:
    confidence_threshold = 0.5

# Interpretability options
st.sidebar.markdown("### Interpretability")
show_cam = st.sidebar.checkbox("Show Class Activation Map", True)
show_confidence = st.sidebar.checkbox("Show Confidence Breakdown", True)

# Clear previous results
if st.sidebar.button("Clear Results"):
    st.session_state.prediction_results = []
    st.rerun()

# Main content
if prediction_mode == "Single Image":
    st.markdown("### 📤 Upload Medical Image for Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Initialize image source
        uploaded_file = None
        
        # Sample image selector
        st.markdown("#### 🔬 Use Sample Images")
        use_sample = st.checkbox("Use sample lung cancer images for testing")
        
        if use_sample:
            sample_type = st.selectbox(
                "Select sample image type:",
                ["cancer", "normal"],
                help="Choose between cancer or normal lung images"
            )
            
            # Get list of sample images
            sample_dir = f"sample_data/lung_cancer_dataset/{sample_type}"
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
                if sample_files:
                    selected_sample = st.selectbox(
                        "Select sample image:",
                        sample_files,
                        help="Choose a specific sample image"
                    )
                    sample_path = os.path.join(sample_dir, selected_sample)
                    
                    if st.button("Load Sample Image", use_container_width=True):
                        uploaded_file = sample_path
                        st.success(f"Loaded sample image: {selected_sample}")
                        st.session_state.current_sample_image = sample_path
                    elif 'current_sample_image' in st.session_state:
                        uploaded_file = st.session_state.current_sample_image
                else:
                    st.warning("No sample images found in the selected category")
            else:
                st.warning("Sample images not available. Please upload your own image.")
        
        if not use_sample:
            st.markdown("#### 📤 Upload Your Own Image")
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a medical image file",
                type=['dcm', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload CT scan, X-ray, or other medical image"
            )
            # Clear sample image if switching to upload mode
            if 'current_sample_image' in st.session_state:
                del st.session_state.current_sample_image
        
        if uploaded_file is not None:
            try:
                # Handle sample image path vs uploaded file
                if isinstance(uploaded_file, str):  # Sample image path
                    tmp_file_path = uploaded_file
                    file_name = os.path.basename(uploaded_file)
                else:  # Uploaded file
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name
                    file_name = uploaded_file.name
                
                # Load and preprocess image
                with st.spinner("Processing image..."):
                    processed_image = preprocessor.load_and_preprocess(tmp_file_path, enhance_contrast=True)
                    
                # Display original and processed images
                st.markdown("#### Original Image")
                
                # Load original for display
                if file_name.lower().endswith('.dcm'):
                    original_image = preprocessor.load_dicom(tmp_file_path)
                else:
                    original_image = preprocessor.load_standard_image(tmp_file_path)
                
                # Display original
                if len(original_image.shape) == 2:
                    st.image(original_image, caption="Original Image", use_column_width=True, clamp=True)
                else:
                    st.image(original_image, caption="Original Image", use_column_width=True)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                uploaded_file = None
    
    with col2:
        if uploaded_file is not None:
            st.markdown("#### Processed Image (Model Input)")
            
            # Display processed image
            display_image = processed_image.copy()
            if display_image.max() <= 1.0:
                display_image = (display_image * 255).astype(np.uint8)
            
            st.image(display_image, caption="Processed Image (224x224)", use_column_width=True)
            
            # Prediction button
            if st.button("🔍 Make Prediction", type="primary", use_container_width=True):
                with st.spinner("Making prediction..."):
                    try:
                        # Prepare image for prediction
                        image_batch = np.expand_dims(processed_image, axis=0)
                        
                        # Make prediction
                        if len(class_names) == 2:
                            # Binary classification
                            pred_proba = model.model.predict(image_batch)[0][0]
                            pred_class = int(pred_proba > confidence_threshold)
                            confidence = pred_proba if pred_class == 1 else (1 - pred_proba)
                        else:
                            # Multi-class classification
                            pred_proba_all = model.model.predict(image_batch)[0]
                            pred_class = np.argmax(pred_proba_all)
                            confidence = pred_proba_all[pred_class]
                            pred_proba = pred_proba_all
                        
                        # Store prediction result
                        prediction_result = {
                            'image': processed_image,
                            'original_image': original_image,
                            'filename': uploaded_file.name,
                            'predicted_class': pred_class,
                            'confidence': confidence,
                            'probabilities': pred_proba,
                            'timestamp': pd.Timestamp.now()
                        }
                        
                        st.session_state.prediction_results.append(prediction_result)
                        st.success("Prediction completed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

else:  # Batch Images
    st.markdown("### 📤 Upload Multiple Images for Batch Prediction")
    
    uploaded_files = st.file_uploader(
        "Choose multiple medical image files",
        type=['dcm', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload multiple CT scans, X-rays, or other medical images"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files uploaded**")
        
        # Show file list
        with st.expander("View uploaded files"):
            for i, file in enumerate(uploaded_files):
                st.text(f"{i+1}. {file.name}")
        
        if st.button("🔍 Process All Images", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            batch_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name
                    
                    # Load and preprocess image
                    processed_image = preprocessor.load_and_preprocess(tmp_file_path, enhance_contrast=True)
                    
                    # Load original for display
                    if uploaded_file.name.lower().endswith('.dcm'):
                        original_image = preprocessor.load_dicom(tmp_file_path)
                    else:
                        original_image = preprocessor.load_standard_image(tmp_file_path)
                    
                    # Make prediction
                    image_batch = np.expand_dims(processed_image, axis=0)
                    
                    if len(class_names) == 2:
                        # Binary classification
                        pred_proba = model.model.predict(image_batch, verbose=0)[0][0]
                        pred_class = int(pred_proba > confidence_threshold)
                        confidence = pred_proba if pred_class == 1 else (1 - pred_proba)
                    else:
                        # Multi-class classification
                        pred_proba_all = model.model.predict(image_batch, verbose=0)[0]
                        pred_class = np.argmax(pred_proba_all)
                        confidence = pred_proba_all[pred_class]
                        pred_proba = pred_proba_all
                    
                    # Store result
                    prediction_result = {
                        'image': processed_image,
                        'original_image': original_image,
                        'filename': uploaded_file.name,
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'probabilities': pred_proba,
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    batch_results.append(prediction_result)
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Add batch results to session state
            st.session_state.prediction_results.extend(batch_results)
            
            status_text.text("Batch processing completed!")
            st.success(f"Successfully processed {len(batch_results)} out of {len(uploaded_files)} images")
            st.rerun()

# Display prediction results
if st.session_state.prediction_results:
    st.markdown("---")
    st.markdown("### 🎯 Prediction Results")
    
    # Summary statistics
    total_predictions = len(st.session_state.prediction_results)
    
    if len(class_names) == 2:
        cancer_predictions = sum(1 for r in st.session_state.prediction_results if r['predicted_class'] == 1)
        normal_predictions = total_predictions - cancer_predictions
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric("Cancer Detected", cancer_predictions)
        with col3:
            st.metric("Normal", normal_predictions)
        with col4:
            avg_confidence = np.mean([r['confidence'] for r in st.session_state.prediction_results])
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    else:
        # Multi-class summary
        class_counts = {}
        for class_name in class_names:
            class_counts[class_name] = sum(1 for r in st.session_state.prediction_results 
                                         if class_names[r['predicted_class']] == class_name)
        
        cols = st.columns(len(class_names) + 1)
        with cols[0]:
            st.metric("Total Predictions", total_predictions)
        
        for i, (class_name, count) in enumerate(class_counts.items()):
            with cols[i + 1]:
                st.metric(class_name, count)
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    
    results_data = []
    for i, result in enumerate(st.session_state.prediction_results):
        results_data.append({
            "ID": i + 1,
            "Filename": result['filename'],
            "Predicted Class": class_names[result['predicted_class']],
            "Confidence": f"{result['confidence']:.3f}",
            "Timestamp": result['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Add color coding for predictions
    def highlight_predictions(row):
        if len(class_names) == 2:
            if row['Predicted Class'] == 'Cancer':
                return ['background-color: #ffcccc'] * len(row)  # Light red for cancer
            else:
                return ['background-color: #ccffcc'] * len(row)  # Light green for normal
        return [''] * len(row)
    
    styled_df = results_df.style.apply(highlight_predictions, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # Individual result visualization
    st.markdown("### 🖼️ Individual Predictions")
    
    # Select result to view in detail
    selected_idx = st.selectbox(
        "Select prediction to view in detail:",
        range(len(st.session_state.prediction_results)),
        format_func=lambda x: f"#{x+1}: {st.session_state.prediction_results[x]['filename']} - {class_names[st.session_state.prediction_results[x]['predicted_class']]}"
    )
    
    if selected_idx is not None:
        selected_result = st.session_state.prediction_results[selected_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            orig_img = selected_result['original_image']
            if len(orig_img.shape) == 2:
                st.image(orig_img, caption="Original Medical Image", use_column_width=True, clamp=True)
            else:
                st.image(orig_img, caption="Original Medical Image", use_column_width=True)
        
        with col2:
            st.markdown("#### Processed Image")
            proc_img = selected_result['image']
            if proc_img.max() <= 1.0:
                display_proc = (proc_img * 255).astype(np.uint8)
            else:
                display_proc = proc_img
            st.image(display_proc, caption="Processed Image", use_column_width=True)
        
        # Prediction details
        st.markdown("#### Prediction Details")
        
        pred_class = selected_result['predicted_class']
        confidence = selected_result['confidence']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Class", class_names[pred_class])
        with col2:
            st.metric("Confidence", f"{confidence:.3f}")
        with col3:
            # Risk level based on confidence and prediction
            if len(class_names) == 2 and pred_class == 1:  # Cancer predicted
                if confidence > 0.8:
                    risk_level = "High"
                    risk_color = "red"
                elif confidence > 0.6:
                    risk_level = "Medium"
                    risk_color = "orange"
                else:
                    risk_level = "Low"
                    risk_color = "yellow"
                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
            else:
                st.markdown("**Risk Level:** :green[Low]")
        
        # Confidence breakdown
        if show_confidence:
            st.markdown("#### Confidence Breakdown")
            
            if len(class_names) == 2:
                # Binary classification
                prob_cancer = selected_result['probabilities']
                prob_normal = 1 - prob_cancer
                
                conf_data = {
                    "Class": ["Normal", "Cancer"],
                    "Probability": [prob_normal, prob_cancer],
                    "Percentage": [f"{prob_normal*100:.1f}%", f"{prob_cancer*100:.1f}%"]
                }
                
                conf_df = pd.DataFrame(conf_data)
                st.dataframe(conf_df, use_container_width=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(conf_data["Class"], conf_data["Probability"], 
                             color=['lightgreen', 'lightcoral'])
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Confidence')
                ax.set_ylim(0, 1)
                
                # Add percentage labels on bars
                for bar, pct in zip(bars, conf_data["Percentage"]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           pct, ha='center', va='bottom')
                
                st.pyplot(fig)
                
            else:
                # Multi-class classification
                probabilities = selected_result['probabilities']
                
                conf_data = {
                    "Class": class_names,
                    "Probability": probabilities,
                    "Percentage": [f"{p*100:.1f}%" for p in probabilities]
                }
                
                conf_df = pd.DataFrame(conf_data)
                conf_df = conf_df.sort_values('Probability', ascending=False)
                st.dataframe(conf_df, use_container_width=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(conf_df["Class"], conf_df["Probability"])
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Confidence')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                
                # Add percentage labels on bars
                for bar, pct in zip(bars, conf_df["Percentage"]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           pct, ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Class Activation Map
        if show_cam:
            st.markdown("#### Class Activation Map (CAM)")
            st.info("CAM shows which regions of the image the model focused on for its prediction")
            
            if st.button("Generate CAM", key=f"cam_{selected_idx}"):
                with st.spinner("Generating Class Activation Map..."):
                    try:
                        fig_cam, heatmap = viz.create_class_activation_map(
                            model.model, selected_result['image'], pred_class
                        )
                        st.pyplot(fig_cam)
                        
                        # CAM interpretation
                        st.markdown("##### CAM Interpretation")
                        if len(class_names) == 2:
                            if pred_class == 1:  # Cancer
                                st.warning("""
                                **Red/Yellow regions** indicate areas the model considers suspicious for cancer.
                                These regions contributed most to the cancer classification decision.
                                """)
                            else:  # Normal
                                st.success("""
                                **Red/Yellow regions** show areas the model examined but found to be normal.
                                The overall pattern supports a normal classification.
                                """)
                        else:
                            st.info(f"""
                            **Red/Yellow regions** show areas most relevant for classifying this image as 
                            **{class_names[pred_class]}**.
                            """)
                        
                    except Exception as e:
                        st.error(f"Error generating CAM: {str(e)}")

# Export functionality
if st.session_state.prediction_results:
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export CSV Report", use_container_width=True):
            # Create detailed CSV report
            export_data = []
            for i, result in enumerate(st.session_state.prediction_results):
                row = {
                    "ID": i + 1,
                    "Filename": result['filename'],
                    "Predicted_Class": class_names[result['predicted_class']],
                    "Predicted_Class_Index": result['predicted_class'],
                    "Confidence": result['confidence'],
                    "Timestamp": result['timestamp']
                }
                
                # Add probability scores
                if len(class_names) == 2:
                    row["Probability_Normal"] = 1 - result['probabilities']
                    row["Probability_Cancer"] = result['probabilities']
                else:
                    for j, class_name in enumerate(class_names):
                        row[f"Probability_{class_name}"] = result['probabilities'][j]
                
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            csv_data = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"lung_cancer_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Summary Report", use_container_width=True):
            # Create summary report
            summary_report = f"""
# Lung Cancer Detection - Prediction Summary Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Predictions: {len(st.session_state.prediction_results)}
"""
            
            if len(class_names) == 2:
                cancer_count = sum(1 for r in st.session_state.prediction_results if r['predicted_class'] == 1)
                normal_count = len(st.session_state.prediction_results) - cancer_count
                
                summary_report += f"""
- Cancer Detected: {cancer_count} ({cancer_count/len(st.session_state.prediction_results)*100:.1f}%)
- Normal: {normal_count} ({normal_count/len(st.session_state.prediction_results)*100:.1f}%)
"""
            
            avg_confidence = np.mean([r['confidence'] for r in st.session_state.prediction_results])
            summary_report += f"""
- Average Confidence: {avg_confidence:.3f}

## Model Information
- Architecture: {model.architecture.upper()}
- Classes: {', '.join(class_names)}
- Input Size: 224x224 pixels

## Detailed Results
"""
            
            for i, result in enumerate(st.session_state.prediction_results):
                summary_report += f"""
### Prediction {i+1}
- **File:** {result['filename']}
- **Prediction:** {class_names[result['predicted_class']]}
- **Confidence:** {result['confidence']:.3f}
- **Timestamp:** {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            st.download_button(
                label="Download Summary",
                data=summary_report,
                file_name=f"prediction_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    with col3:
        st.button("Save Images", disabled=True, use_container_width=True)
        st.caption("Image export would be implemented in production")

# Clinical recommendations
if st.session_state.prediction_results and len(class_names) == 2:
    cancer_predictions = [r for r in st.session_state.prediction_results if r['predicted_class'] == 1]
    
    if cancer_predictions:
        st.markdown("---")
        st.markdown("### ⚕️ Clinical Recommendations")
        
        high_confidence_cancer = [r for r in cancer_predictions if r['confidence'] > 0.8]
        medium_confidence_cancer = [r for r in cancer_predictions if 0.6 <= r['confidence'] <= 0.8]
        low_confidence_cancer = [r for r in cancer_predictions if r['confidence'] < 0.6]
        
        if high_confidence_cancer:
            st.error(f"""
            **🚨 High Priority Cases ({len(high_confidence_cancer)}):**
            Strong indication of cancer detected. Immediate clinical review recommended.
            """)
        
        if medium_confidence_cancer:
            st.warning(f"""
            **⚠️ Medium Priority Cases ({len(medium_confidence_cancer)}):**
            Possible cancer indication. Clinical evaluation advised.
            """)
        
        if low_confidence_cancer:
            st.info(f"""
            **ℹ️ Low Priority Cases ({len(low_confidence_cancer)}):**
            Weak cancer indication. Consider follow-up imaging or monitoring.
            """)
        
        st.markdown("""
        **Important Note:** 
        - This AI system is designed to assist medical professionals, not replace clinical judgment
        - All positive predictions should be verified by qualified radiologists
        - Consider patient history, symptoms, and other clinical factors
        - False positives and false negatives are possible with any AI system
        """)

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to Evaluation", use_container_width=True):
        st.switch_page("pages/3_Model_Evaluation.py")
with col2:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")
