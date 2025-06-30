import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.evaluation import MedicalModelEvaluator
from utils.visualization import MedicalVisualization
import seaborn as sns

st.set_page_config(
    page_title="Model Evaluation - Lung Cancer Detection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Evaluation and Performance Analysis")

# Initialize session state variables if needed
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = None

# Check if model is trained
if not st.session_state.model_trained or st.session_state.trained_model is None:
    st.error("No trained model found. Please train a model first.")
    if st.button("Go to Model Training"):
        st.switch_page("pages/2_Model_Training.py")
    st.stop()

# Check if data is available
if st.session_state.data_split is None:
    st.error("No dataset found. Please upload and process data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Get data from session state
model = st.session_state.trained_model
data_split = st.session_state.data_split
class_names = st.session_state.class_names

X_test = data_split['X_test']
y_test = data_split['y_test']
X_val = data_split['X_val']
y_val = data_split['y_val']

# Initialize evaluator and visualizer
evaluator = MedicalModelEvaluator(class_names)
viz = MedicalVisualization(class_names)

# Sidebar for evaluation options
st.sidebar.markdown("### Evaluation Options")
eval_dataset = st.sidebar.radio(
    "Evaluate on:",
    ["Test Set", "Validation Set"],
    help="Choose which dataset to evaluate the model on"
)

show_detailed = st.sidebar.checkbox("Show Detailed Analysis", True)
show_predictions = st.sidebar.checkbox("Show Sample Predictions", True)
show_interpretability = st.sidebar.checkbox("Show Model Interpretability", True)

# Select dataset based on choice
if eval_dataset == "Test Set":
    X_eval = X_test
    y_eval = y_test
    dataset_name = "Test"
else:
    X_eval = X_val
    y_eval = y_val
    dataset_name = "Validation"

# Perform evaluation
if st.session_state.model_performance is None or st.button("🔄 Run Evaluation", type="primary"):
    with st.spinner(f"Evaluating model on {dataset_name.lower()} set..."):
        try:
            # Evaluate model
            results = evaluator.evaluate_model(model.model, X_eval, y_eval)
            
            # Store results in session state
            st.session_state.model_performance = results
            
            st.success(f"Model evaluation completed on {dataset_name.lower()} set!")
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.stop()

# Get results
results = st.session_state.model_performance

# Main metrics overview
st.markdown(f"### 🎯 Model Performance on {dataset_name} Set")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{results['accuracy']:.3f}")
with col2:
    st.metric("Precision", f"{results['precision']:.3f}")
with col3:
    st.metric("Recall", f"{results['recall']:.3f}")
with col4:
    st.metric("F1-Score", f"{results['f1_score']:.3f}")

if len(class_names) == 2:  # Binary classification
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sensitivity", f"{results['sensitivity']:.3f}")
    with col2:
        st.metric("Specificity", f"{results['specificity']:.3f}")
    with col3:
        st.metric("AUC-ROC", f"{results['auc']:.3f}")

# Medical metrics summary
st.markdown("### 🏥 Medical Performance Metrics")
medical_metrics_df = evaluator.create_medical_metrics_summary(results)
st.dataframe(medical_metrics_df, use_container_width=True)

# Visualizations
st.markdown("---")
st.markdown("### 📈 Performance Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Classification Report", "Predictions"])

with tab1:
    st.markdown("#### Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Raw Counts**")
        fig_cm = evaluator.plot_confusion_matrix(results['confusion_matrix'], normalize=False)
        st.pyplot(fig_cm)
    
    with col2:
        st.markdown("**Normalized**")
        fig_cm_norm = evaluator.plot_confusion_matrix(results['confusion_matrix'], normalize=True)
        st.pyplot(fig_cm_norm)
    
    # Confusion matrix insights
    cm = results['confusion_matrix']
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown("#### Confusion Matrix Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**True Positives (Cancer correctly identified):**")
            st.metric("TP", tp)
            st.markdown("**False Negatives (Cancer missed):**")
            st.metric("FN", fn)
        
        with col2:
            st.markdown("**True Negatives (Normal correctly identified):**")
            st.metric("TN", tn)
            st.markdown("**False Positives (Normal misclassified as Cancer):**")
            st.metric("FP", fp)
        
        # Clinical interpretation
        st.markdown("#### Clinical Interpretation")
        st.info(f"""
        **Clinical Impact Analysis:**
        - **Missed Cancer Cases (False Negatives):** {fn} cases
        - **False Alarms (False Positives):** {fp} cases
        - **Sensitivity (Cancer Detection Rate):** {results['sensitivity']:.1%}
        - **Specificity (Normal Identification Rate):** {results['specificity']:.1%}
        
        In medical applications, false negatives (missed cancer) are typically more concerning than false positives.
        """)

with tab2:
    st.markdown("#### ROC Curve Analysis")
    
    if len(class_names) == 2:
        fig_roc = evaluator.plot_roc_curve(results)
        st.pyplot(fig_roc)
        
        # ROC interpretation
        st.markdown("#### ROC Curve Interpretation")
        auc_value = results['auc']
        
        if auc_value >= 0.9:
            auc_quality = "Excellent"
            auc_color = "green"
        elif auc_value >= 0.8:
            auc_quality = "Good"
            auc_color = "blue"
        elif auc_value >= 0.7:
            auc_quality = "Fair"
            auc_color = "orange"
        else:
            auc_quality = "Poor"
            auc_color = "red"
        
        st.markdown(f"**AUC Score: {auc_value:.3f}** - :{auc_color}[{auc_quality}]")
        
        # Precision-Recall curve
        if results['pr_auc'] is not None:
            st.markdown("#### Precision-Recall Curve")
            fig_pr = evaluator.plot_precision_recall_curve(results)
            st.pyplot(fig_pr)
    else:
        fig_roc = evaluator.plot_roc_curve(results)
        st.pyplot(fig_roc)

with tab3:
    st.markdown("#### Detailed Classification Report")
    
    # Convert classification report to DataFrame
    class_report_df = evaluator.generate_classification_report_df(results['classification_report'])
    
    # Style the DataFrame
    styled_df = class_report_df.style.background_gradient(
        subset=['Precision', 'Recall', 'F1-Score'], 
        cmap='RdYlGn'
    ).format({
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1-Score': '{:.3f}',
        'Support': '{:.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Per-class analysis
    st.markdown("#### Per-Class Performance Analysis")
    
    for i, class_name in enumerate(class_names):
        if class_name in results['classification_report']:
            class_metrics = results['classification_report'][class_name]
            
            with st.expander(f"📋 {class_name} Class Analysis"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Precision", f"{class_metrics['precision']:.3f}")
                    st.caption("Of predicted positives, how many were correct?")
                
                with col2:
                    st.metric("Recall", f"{class_metrics['recall']:.3f}")
                    st.caption("Of actual positives, how many were found?")
                
                with col3:
                    st.metric("F1-Score", f"{class_metrics['f1-score']:.3f}")
                    st.caption("Harmonic mean of precision and recall")
                
                st.markdown(f"**Support:** {class_metrics['support']} samples")

with tab4:
    if show_predictions:
        st.markdown("#### Sample Predictions Analysis")
        
        # Number of samples to show
        num_samples = st.slider("Number of samples to display", 4, 16, 8)
        
        # Select random samples
        sample_indices = np.random.choice(len(X_eval), size=min(num_samples, len(X_eval)), replace=False)
        
        sample_images = X_eval[sample_indices]
        sample_true = y_eval[sample_indices]
        sample_pred = results['y_pred'][sample_indices]
        
        if len(results['y_pred_proba'].shape) == 1:  # Binary classification
            sample_probs = results['y_pred_proba'][sample_indices]
        else:  # Multi-class
            sample_probs = results['y_pred_proba'][sample_indices]
        
        # Visualize predictions
        fig_pred = viz.visualize_predictions(
            sample_images, sample_true, sample_pred, sample_probs,
            num_samples=num_samples
        )
        st.pyplot(fig_pred)
        
        # Prediction confidence distribution
        st.markdown("#### Prediction Confidence Distribution")
        fig_conf = viz.plot_prediction_confidence_distribution(
            results['y_pred_proba'], results['y_pred']
        )
        st.pyplot(fig_conf)

# Model interpretability section
if show_interpretability:
    st.markdown("---")
    st.markdown("### 🔍 Model Interpretability")
    
    st.markdown("#### Class Activation Maps (CAM)")
    st.info("Select an image to generate Class Activation Map for model interpretability")
    
    # Select image for CAM
    cam_sample_idx = st.selectbox(
        "Select sample for CAM analysis",
        range(min(10, len(X_eval))),
        format_func=lambda x: f"Sample {x+1} - True: {class_names[y_eval[x]]}, Pred: {class_names[results['y_pred'][x]]}"
    )
    
    if st.button("Generate Class Activation Map"):
        with st.spinner("Generating CAM..."):
            try:
                sample_image = X_eval[cam_sample_idx]
                predicted_class = results['y_pred'][cam_sample_idx]
                
                fig_cam, heatmap = viz.create_class_activation_map(
                    model.model, sample_image, predicted_class
                )
                st.pyplot(fig_cam)
                
                # CAM interpretation
                st.markdown("#### CAM Interpretation")
                st.success("""
                **Class Activation Map shows:**
                - **Red/Yellow areas:** Regions most important for the prediction
                - **Blue/Dark areas:** Regions less relevant for the decision
                - This helps understand what the model is "looking at" when making predictions
                """)
                
            except Exception as e:
                st.error(f"Error generating CAM: {str(e)}")
    
    # Feature maps visualization
    st.markdown("#### Feature Maps Visualization")
    if st.button("Show Feature Maps"):
        with st.spinner("Generating feature maps..."):
            try:
                sample_image = X_eval[0]  # Use first sample
                fig_features = viz.visualize_feature_maps(model.model, sample_image)
                st.pyplot(fig_features)
                
                st.info("""
                **Feature Maps show:**
                - How different layers of the CNN process the input image
                - Early layers detect basic features (edges, textures)
                - Deeper layers detect more complex patterns
                """)
                
            except Exception as e:
                st.error(f"Error generating feature maps: {str(e)}")

# Training history comparison
if st.session_state.training_history is not None:
    st.markdown("---")
    st.markdown("### 📈 Training vs. Evaluation Performance")
    
    history = st.session_state.training_history
    
    # Compare final training metrics with evaluation metrics
    comparison_data = []
    
    if 'val_accuracy' in history:
        comparison_data.append({
            "Metric": "Accuracy",
            "Final Training": f"{history['accuracy'][-1]:.3f}",
            "Final Validation": f"{history['val_accuracy'][-1]:.3f}",
            f"Current {dataset_name}": f"{results['accuracy']:.3f}"
        })
    
    if 'val_loss' in history:
        comparison_data.append({
            "Metric": "Loss",
            "Final Training": f"{history['loss'][-1]:.3f}",
            "Final Validation": f"{history['val_loss'][-1]:.3f}",
            f"Current {dataset_name}": "N/A"
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Plot training history
        fig_history = viz.plot_training_history(history)
        st.pyplot(fig_history)

# Model recommendations
st.markdown("---")
st.markdown("### 💡 Model Performance Recommendations")

# Generate recommendations based on performance
recommendations = []

if results['accuracy'] < 0.8:
    recommendations.append("🔸 Consider collecting more training data or trying different architectures")

if len(class_names) == 2:
    if results['sensitivity'] < 0.8:
        recommendations.append("🔸 Low sensitivity - consider adjusting classification threshold or addressing class imbalance")
    
    if results['specificity'] < 0.8:
        recommendations.append("🔸 Low specificity - model may be too aggressive in cancer detection")
    
    if results['auc'] < 0.8:
        recommendations.append("🔸 Consider feature engineering or ensemble methods to improve AUC")

if abs(results['precision'] - results['recall']) > 0.1:
    recommendations.append("🔸 Imbalanced precision/recall - consider adjusting class weights or threshold")

if not recommendations:
    recommendations.append("✅ Model performance looks good! Consider validating on external datasets")

for rec in recommendations:
    st.markdown(rec)

# Export results
st.markdown("---")
st.markdown("### 💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Evaluation Report", use_container_width=True):
        # Create comprehensive report
        report_data = {
            "Model Performance Summary": medical_metrics_df,
            "Classification Report": class_report_df,
            "Confusion Matrix": pd.DataFrame(results['confusion_matrix'], 
                                           columns=class_names, index=class_names)
        }
        
        st.success("Report data prepared for export")
        st.json(results, expanded=False)

with col2:
    st.button("Save Evaluation Plots", disabled=True, use_container_width=True)
    st.caption("Plot saving would be implemented in production")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to Training", use_container_width=True):
        st.switch_page("pages/2_Model_Training.py")
with col2:
    if st.button("Next: Prediction →", type="primary", use_container_width=True):
        st.switch_page("pages/4_Prediction.py")
