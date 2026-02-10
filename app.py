import streamlit as st
import os
import numpy as np
from PIL import Image
import cv2

# Configure page - clean, modern design
st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with Chocolate Truffle color scheme
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #38240D 0%, #713600 50%, #C05800 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    .hero-title {
        color: #FDFBD4;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .hero-subtitle {
        color: rgba(253,251,212,0.9);
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: #FDFBD4;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(56,36,13,0.12);
        border: 1px solid #C05800;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(56,36,13,0.18);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #713600;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #38240D;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #713600;
        font-size: 0.95rem;
    }
    
    /* Stats section */
    .stats-container {
        background: #FDFBD4;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38240D;
    }
    
    .stat-label {
        color: #713600;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #713600 0%, #C05800 100%);
        color: #FDFBD4;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(113,54,0,0.4);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Upload area styling */
    .upload-section {
        background: #FDFBD4;
        border: 2px dashed #C05800;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #FDFBD4;
        border-left: 4px solid #C05800;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-top: 2rem;
        color: #38240D;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #FDFBD4;
        color: #38240D;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #713600 !important;
        color: #FDFBD4 !important;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(135deg, #713600 0%, #C05800 100%);
        color: #FDFBD4;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    .success-box h3 {
        margin: 0;
        color: #FDFBD4;
    }
    
    .success-box p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #FDFBD4;
        color: #38240D;
        border: 1px solid #C05800;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_images' not in st.session_state:
    st.session_state.analyzed_images = []

# Hero Section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Lung Cancer Detection AI</h1>
    <p class="hero-subtitle">Advanced deep learning analysis for medical imaging</p>
</div>
""", unsafe_allow_html=True)

# Main tabs for navigation
tab1, tab2, tab3 = st.tabs(["Analyze Image", "About the Technology", "Sample Images"])

with tab1:
    st.markdown("### Upload a Medical Image for Analysis")
    st.markdown("Upload a chest X-ray or CT scan image to analyze for potential lung abnormalities.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'dcm'],
            help="Supported formats: PNG, JPG, JPEG, BMP, DICOM"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Process image
                    img_array = np.array(image.convert('RGB'))
                    img_resized = cv2.resize(img_array, (224, 224))
                    
                    # Calculate image statistics for demo analysis
                    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                    mean_intensity = np.mean(gray)
                    std_intensity = np.std(gray)
                    
                    # Demo analysis based on image characteristics
                    import time
                    time.sleep(1.5)  # Simulate processing
                    
                    st.session_state.analysis_complete = True
                    st.session_state.analysis_result = {
                        'mean_intensity': mean_intensity,
                        'std_intensity': std_intensity,
                        'image': img_resized
                    }
                    st.rerun()
    
    with col2:
        if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
            result = st.session_state.analysis_result
            
            st.markdown("### Analysis Results")
            
            # Display analysis metrics
            st.markdown("""
            <div class="success-box">
                <h3>Analysis Complete</h3>
                <p>Image successfully processed</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Image quality metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Image Quality", "Good" if result['std_intensity'] > 30 else "Fair")
            with col_b:
                st.metric("Contrast Level", f"{result['std_intensity']:.1f}")
            
            # Findings (demo mode)
            st.markdown("#### Observations")
            st.info("""
            **Demo Mode Results:**
            
            This is a demonstration of the analysis interface. In production with a trained model:
            
            - AI would classify the image as Normal or showing signs of Cancer
            - Confidence scores would be provided
            - Affected regions would be highlighted
            - Detailed medical metrics would be generated
            
            To enable full AI analysis, the model needs to be trained on the Model Training page.
            """)
            
            if st.button("Clear Analysis", use_container_width=True):
                st.session_state.analysis_complete = False
                st.rerun()
        else:
            st.markdown("### How It Works")
            st.markdown("""
            1. **Upload** - Select a chest X-ray or CT scan image
            2. **Analyze** - Our AI processes the image
            3. **Results** - View detailed analysis and findings
            """)
            
            st.markdown("### Supported Image Types")
            st.markdown("""
            - Chest X-rays (PA/AP views)
            - CT scan slices
            - DICOM format images
            - Standard image formats (PNG, JPG)
            """)

with tab2:
    st.markdown("### About Our Technology")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Deep Learning</div>
            <div class="feature-desc">
                Powered by state-of-the-art Convolutional Neural Networks 
                trained on medical imaging datasets.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">High Accuracy</div>
            <div class="feature-desc">
                Transfer learning with ResNet50, DenseNet121, and 
                EfficientNet architectures for optimal performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Fast Analysis</div>
            <div class="feature-desc">
                Get results in seconds with our optimized 
                inference pipeline and cloud infrastructure.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Model Architectures")
    st.markdown("""
    Our system supports multiple pre-trained model architectures:
    
    | Architecture | Parameters | Best For |
    |-------------|-----------|----------|
    | ResNet50 | 25.6M | General classification |
    | DenseNet121 | 8M | Feature reuse |
    | EfficientNetB0 | 5.3M | Balanced performance |
    """)

with tab3:
    st.markdown("### Sample Lung Images")
    st.markdown("Browse sample images from our training dataset.")
    
    sample_dir = "sample_data/lung_cancer_dataset"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Normal Lung Images")
        normal_dir = os.path.join(sample_dir, "normal")
        if os.path.exists(normal_dir):
            normal_files = [f for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:4]
            if normal_files:
                cols = st.columns(2)
                for i, f in enumerate(normal_files):
                    with cols[i % 2]:
                        img_path = os.path.join(normal_dir, f)
                        img = Image.open(img_path)
                        st.image(img, caption=f"Normal - {f}")
            else:
                st.info("No normal sample images available")
        else:
            st.info("Sample images not found")
    
    with col2:
        st.markdown("#### Cancer Lung Images")
        cancer_dir = os.path.join(sample_dir, "cancer")
        if os.path.exists(cancer_dir):
            cancer_files = [f for f in os.listdir(cancer_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][:4]
            if cancer_files:
                cols = st.columns(2)
                for i, f in enumerate(cancer_files):
                    with cols[i % 2]:
                        img_path = os.path.join(cancer_dir, f)
                        img = Image.open(img_path)
                        st.image(img, caption=f"Cancer - {f}")
            else:
                st.info("No cancer sample images available")
        else:
            st.info("Sample images not found")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This application is for research and educational 
    purposes only. It should not be used as a substitute for professional medical diagnosis. 
    Always consult with qualified healthcare providers for medical advice.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #713600; padding: 1rem;">
    <p>Lung Cancer Detection AI | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
