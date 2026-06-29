import streamlit as st
import streamlit.components.v1 as components
import os
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
from utils.visualization import MedicalVisualization
from utils.model_state import ensure_model_loaded

# Configure page - clean, modern, flat design
st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Flat theme — solid colours, no gradients. (Palette: chocolate truffle.)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }

    /* Feature cards — flat, bordered, no gradient */
    .feature-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e7d9b0;
        box-shadow: 0 1px 3px rgba(56,36,13,0.08);
        height: 100%;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #38240D;
        margin-bottom: 0.5rem;
    }
    .feature-desc { color: #713600; font-size: 0.95rem; }

    /* Status banners — solid colours */
    .status-ok {
        background: #1b5e20; color: #FDFBD4;
        padding: 0.85rem 1.1rem; border-radius: 10px; margin: 0.5rem 0 1rem 0;
        font-weight: 600;
    }
    .status-warn {
        background: #8a4b00; color: #FDFBD4;
        padding: 0.85rem 1.1rem; border-radius: 10px; margin: 0.5rem 0 1rem 0;
        font-weight: 500;
    }

    /* Buttons — solid, colour shift on hover (no gradient) */
    .stButton > button {
        background: #713600;
        color: #FDFBD4;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #C05800; color: #FDFBD4; }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .disclaimer {
        background: #ffffff;
        border-left: 4px solid #C05800;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-top: 2rem;
        color: #38240D;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        color: #38240D;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #713600 !important;
        color: #FDFBD4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3D hero — self-contained Three.js scene (rotating point-cloud + wireframe).
# Solid dark background, no gradient. Title overlaid on the canvas.
# ---------------------------------------------------------------------------
HERO_HTML = """
<div id="hero">
  <canvas id="scene"></canvas>
  <div id="overlay">
    <h1>Lung Cancer Detection AI</h1>
    <p>Deep-learning analysis for medical imaging</p>
  </div>
</div>
<style>
  html, body { margin: 0; }
  #hero {
    position: relative; width: 100%; height: 340px;
    background: #1c1306; border-radius: 16px; overflow: hidden;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  #scene { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  #overlay {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center;
    pointer-events: none;
  }
  #overlay h1 {
    color: #FDFBD4; font-size: 2.6rem; font-weight: 700; margin: 0;
    text-shadow: 0 2px 12px rgba(0,0,0,0.6);
  }
  #overlay p {
    color: rgba(253,251,212,0.88); font-size: 1.1rem; margin: 0.5rem 0 0 0;
    text-shadow: 0 1px 8px rgba(0,0,0,0.6);
  }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
  (function () {
    if (typeof THREE === "undefined") { return; }  // graceful: overlay still shows
    const canvas = document.getElementById("scene");
    const hero = document.getElementById("hero");

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.z = 4;

    const group = new THREE.Group();
    scene.add(group);

    // Wireframe icosahedron (accent orange)
    const geo = new THREE.IcosahedronGeometry(1.3, 1);
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0xC05800 })
    );
    group.add(wire);

    // Point-cloud shell (cream) — evokes a volumetric scan
    const pts = [];
    for (let i = 0; i < 900; i++) {
      const u = Math.random(), v = Math.random();
      const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1);
      const r = 1.55;
      pts.push(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi)
      );
    }
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    const points = new THREE.Points(
      pgeo,
      new THREE.PointsMaterial({ color: 0xFDFBD4, size: 0.03 })
    );
    group.add(points);

    function resize() {
      const w = hero.clientWidth, h = hero.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    window.addEventListener("resize", resize);

    function animate() {
      requestAnimationFrame(animate);
      group.rotation.y += 0.004;
      group.rotation.x += 0.0015;
      renderer.render(scene, camera);
    }
    animate();
  })();
</script>
"""
components.html(HERO_HTML, height=360)


# ---------------------------------------------------------------------------
# Real image diagnostics (computed from pixels — no fabricated "AI" output).
# ---------------------------------------------------------------------------
def compute_image_diagnostics(gray):
    """Genuine image-quality metrics derived from the pixels."""
    mean_intensity = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # focus measure
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))  # Shannon entropy (bits)
    return {
        "mean": mean_intensity,
        "contrast": contrast,
        "sharpness": sharpness,
        "entropy": entropy,
    }


def build_intensity_surface(gray):
    """Interactive 3D surface of pixel intensity for the uploaded scan."""
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(float)
    z = np.flipud(small)  # match image orientation (row 0 at top)
    fig = go.Figure(data=[go.Surface(z=z, colorscale="Cividis", showscale=False)])
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Intensity"),
            aspectratio=dict(x=1, y=1, z=0.45),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Main tabs
tab1, tab2, tab3 = st.tabs(["Analyze Image", "About the Technology", "Sample Images"])

with tab1:
    # Use an in-session model if present, else a committed artifact from train.py
    model_ready = ensure_model_loaded()

    if model_ready:
        st.markdown(
            '<div class="status-ok">AI model loaded — full classification enabled.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-warn">No trained AI model is loaded — showing real image '
            'diagnostics only (not a diagnosis). Train a model on the Model Training page '
            'to enable AI classification.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Go to Model Training"):
            st.switch_page("pages/2_Model_Training.py")

    st.markdown("### Upload a Medical Image for Analysis")
    st.markdown("Upload a chest X-ray or CT scan image to analyze.")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, BMP"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Analyze Image", use_container_width=True):
                img_array = np.array(image)
                img_resized = cv2.resize(img_array, (224, 224))
                gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                st.session_state.analysis_result = {'rgb': img_resized, 'gray': gray}
                st.session_state.analysis_complete = True
                st.rerun()

    with col2:
        if st.session_state.analysis_complete and 'analysis_result' in st.session_state:
            res = st.session_state.analysis_result
            gray = res['gray']
            rgb = res['rgb']

            st.markdown("### Analysis Results")

            # --- AI classification (only when a real model is available) ---
            if model_ready:
                model = st.session_state.trained_model
                class_names = st.session_state.get('class_names', ['normal', 'cancer'])
                proc = rgb.astype('float32') / 255.0
                preds = model.model.predict(np.expand_dims(proc, axis=0), verbose=0)

                if preds.shape[-1] == 1:  # binary sigmoid
                    p_pos = float(preds[0][0])
                    pred_class = int(p_pos > 0.5)
                    confidence = p_pos if pred_class == 1 else (1 - p_pos)
                    prob_per_class = [1 - p_pos, p_pos]
                else:  # multi-class softmax
                    probs = preds[0]
                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class])
                    prob_per_class = list(map(float, probs))

                label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
                cA, cB = st.columns(2)
                with cA:
                    st.metric("AI Prediction", label.capitalize())
                with cB:
                    st.metric("Confidence", f"{confidence:.1%}")

                prob_table = {
                    class_names[i] if i < len(class_names) else str(i): f"{prob_per_class[i]:.1%}"
                    for i in range(len(prob_per_class))
                }
                st.write("**Class probabilities**")
                st.table(prob_table)

                # Real Grad-CAM overlay
                st.markdown("#### Grad-CAM (regions driving the prediction)")
                try:
                    viz = MedicalVisualization(class_names)
                    fig_cam, _ = viz.create_class_activation_map(model.model, proc, pred_class)
                    st.pyplot(fig_cam)
                except Exception as e:
                    st.info(f"Grad-CAM unavailable: {e}")

            # --- Real image diagnostics (always shown) ---
            st.markdown("#### Image Diagnostics")
            diag = compute_image_diagnostics(gray)
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Mean intensity", f"{diag['mean']:.1f}")
                st.metric("Sharpness (focus)", f"{diag['sharpness']:.0f}")
            with d2:
                st.metric("Contrast (std)", f"{diag['contrast']:.1f}")
                st.metric("Entropy (bits)", f"{diag['entropy']:.2f}")

            quality = "Good" if diag['contrast'] > 30 and diag['sharpness'] > 50 else "Fair"
            st.caption(f"Image quality assessment: **{quality}** "
                       "(based on measured contrast and focus).")

            if not model_ready:
                st.info(
                    "These are objective image-quality measurements computed directly "
                    "from the pixels — **not** a cancer diagnosis. Train a model to enable "
                    "AI classification with a Grad-CAM explanation."
                )

            if st.button("Clear Analysis", use_container_width=True):
                st.session_state.analysis_complete = False
                st.rerun()
        else:
            st.markdown("### How It Works")
            st.markdown("""
            1. **Upload** — select a chest X-ray or CT scan image
            2. **Analyze** — measure image diagnostics and (if trained) run the AI model
            3. **Results** — view the prediction, Grad-CAM explanation, and a 3D scan render
            """)

    # 3D scan render — full width, below the columns
    if st.session_state.analysis_complete and 'analysis_result' in st.session_state:
        st.markdown("---")
        st.markdown("### 3D Intensity Surface")
        st.caption("Interactive render of the scan's pixel intensity — peaks are denser/brighter tissue. Drag to rotate.")
        st.plotly_chart(
            build_intensity_surface(st.session_state.analysis_result['gray']),
            use_container_width=True,
        )

with tab2:
    st.markdown("### About Our Technology")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Deep Learning</div>
            <div class="feature-desc">
                Convolutional neural networks with transfer learning from
                ImageNet-pretrained backbones.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Explainable</div>
            <div class="feature-desc">
                Real Grad-CAM heatmaps show which regions of the image drove
                each prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Fast Analysis</div>
            <div class="feature-desc">
                Architecture-correct preprocessing and an optimized inference
                pipeline return results in seconds.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Architectures")
    st.markdown("""
    | Architecture | Parameters | Best For |
    |-------------|-----------|----------|
    | ResNet50 | 25.6M | General classification |
    | DenseNet121 | 8M | Feature reuse |
    | EfficientNetB0 | 5.3M | Balanced performance |

    Each backbone uses its own `preprocess_input`, applied inside the model graph,
    so the pretrained weights see the exact input distribution they expect.
    """)

with tab3:
    st.markdown("### Sample Lung Images")
    st.markdown("Browse sample images from the dataset.")

    sample_dir = "sample_data/lung_cancer_dataset"
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Normal Lung Images")
        normal_dir = os.path.join(sample_dir, "normal")
        if os.path.exists(normal_dir):
            normal_files = [f for f in os.listdir(normal_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))][:4]
            if normal_files:
                cols = st.columns(2)
                for i, f in enumerate(normal_files):
                    with cols[i % 2]:
                        st.image(Image.open(os.path.join(normal_dir, f)),
                                 caption=f"Normal - {f}", use_column_width=True)
            else:
                st.info("No normal sample images available")
        else:
            st.info("Sample images not found")

    with col2:
        st.markdown("#### Cancer Lung Images")
        cancer_dir = os.path.join(sample_dir, "cancer")
        if os.path.exists(cancer_dir):
            cancer_files = [f for f in os.listdir(cancer_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))][:4]
            if cancer_files:
                cols = st.columns(2)
                for i, f in enumerate(cancer_files):
                    with cols[i % 2]:
                        st.image(Image.open(os.path.join(cancer_dir, f)),
                                 caption=f"Cancer - {f}", use_column_width=True)
            else:
                st.info("No cancer sample images available")
        else:
            st.info("Sample images not found")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This application is for research and educational
    purposes only. It is not a medical device and must not be used for diagnosis.
    Always consult qualified healthcare providers for medical advice.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #713600; padding: 1rem;">
    <p>Lung Cancer Detection AI | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
