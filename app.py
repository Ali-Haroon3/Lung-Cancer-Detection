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
# Dark "instrument" theme — near-black, one cyan signal, flat panels.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root{
        --bg:#070b12; --panel:#0e141f; --border:rgba(120,160,200,0.14);
        --text:#e7eef6; --muted:#8aa0b6; --accent:#2ee6ff;
        --ok:#36d399; --warn:#ff9d4d;
    }

    /* Typography (avoid icon containers; no global * selector) */
    html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown,
    p, li, label, input, textarea, select, button, .stCaption {
        font-family:'IBM Plex Sans', -apple-system, "Segoe UI", sans-serif;
    }
    h1, h2, h3, h4, h5, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family:'Sora', sans-serif; letter-spacing:-0.01em; color:var(--text);
    }

    .main > div { padding-top: 0.4rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Feature cards — flat dark, cyan edge on hover */
    .feature-card {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 14px; padding: 1.4rem; height: 100%;
        transition: border-color .2s, box-shadow .2s;
    }
    .feature-card:hover {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent), 0 10px 34px rgba(46,230,255,0.08);
    }
    .feature-title { font-family:'Sora'; font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 0.45rem; }
    .feature-desc { color: var(--muted); font-size: 0.92rem; line-height: 1.55; }

    /* Status banners — flat, accent left edge */
    .status-ok, .status-warn {
        background: var(--panel); border: 1px solid var(--border); border-left-width: 3px;
        padding: 0.8rem 1.1rem; border-radius: 10px; margin: 0.4rem 0 1.1rem 0; font-size: 0.95rem;
    }
    .status-ok { border-left-color: var(--ok); color: #cdebdc; }
    .status-warn { border-left-color: var(--warn); color: #f2dcc2; }

    /* Mono kicker label */
    .kicker {
        font-family:'IBM Plex Mono'; text-transform: uppercase; letter-spacing: 0.18em;
        font-size: 0.72rem; color: var(--accent); margin: 0.2rem 0 0.1rem 0;
    }

    /* Metrics -> instrument tiles */
    [data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 12px; padding: 0.7rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-family:'IBM Plex Mono'; text-transform: uppercase;
        letter-spacing: 0.07em; font-size: 0.7rem; color: var(--muted);
    }
    [data-testid="stMetricValue"] { font-family:'Sora'; color: var(--text); }

    /* Buttons — ghost cyan, fill + glow on hover */
    .stButton > button {
        background: transparent; color: var(--accent);
        border: 1px solid var(--accent); border-radius: 9px;
        padding: 0.55rem 1.4rem; font-weight: 600; transition: all .18s;
    }
    .stButton > button:hover { background: var(--accent); color: #04141a; box-shadow: 0 0 22px rgba(46,230,255,0.35); }
    .stButton > button[kind="primary"] { background: var(--accent); color: #04141a; }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 0 26px rgba(46,230,255,0.45); }

    /* File uploader blends into the dark surface */
    [data-testid="stFileUploader"] section {
        background: var(--panel); border: 1px dashed var(--border); border-radius: 12px;
    }

    /* Tabs — quiet, cyan underline when active */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: var(--muted); border-radius: 8px 8px 0 0;
        padding: 10px 18px; font-family:'IBM Plex Mono'; font-size: 0.84rem; letter-spacing: 0.03em;
    }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }

    .disclaimer {
        background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--warn);
        padding: 1rem 1.1rem; border-radius: 10px; margin-top: 1.5rem; color: var(--muted); font-size: 0.9rem;
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
    <div class="kick">AI-ASSISTED &middot; CHEST IMAGING</div>
    <h1>Lung Cancer Detection AI</h1>
    <p>Deep-learning classification with explainable Grad-CAM</p>
  </div>
</div>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;800&family=IBM+Plex+Mono:wght@500&display=swap');
  html, body { margin: 0; height: 100%; background: #070b12; }
  #hero {
    position: relative; width: 100%; height: 100%;
    background: #070b12; border-radius: 16px; overflow: hidden;
    border: 1px solid rgba(120,160,200,0.14);
    font-family: 'Sora', -apple-system, sans-serif;
  }
  /* single soft light source behind the mesh (atmosphere, not UI gradient) */
  #hero::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 46%, rgba(46,230,255,0.16), rgba(7,11,18,0) 55%);
    pointer-events: none;
  }
  #scene { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  #overlay {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center; pointer-events: none;
  }
  .kick {
    font-family: 'IBM Plex Mono', monospace; color: #2ee6ff;
    letter-spacing: 0.34em; font-size: 0.74rem; font-weight: 500;
    margin-bottom: 0.7rem; opacity: 0; transform: translateY(10px);
  }
  #overlay h1 {
    color: #f3f8ff; font-size: 2.9rem; font-weight: 800; margin: 0; line-height: 1.04;
    text-shadow: 0 2px 30px rgba(46,230,255,0.25); opacity: 0; transform: translateY(14px);
  }
  #overlay p {
    color: rgba(199,217,235,0.9); font-size: 1.05rem; margin: 0.7rem 0 0 0;
    opacity: 0; transform: translateY(14px);
  }
  @media (prefers-reduced-motion: no-preference) {
    .kick { animation: rise .7s ease .05s forwards; }
    #overlay h1 { animation: rise .8s cubic-bezier(.2,.7,.2,1) .18s forwards; }
    #overlay p { animation: rise .8s ease .34s forwards; }
  }
  @media (prefers-reduced-motion: reduce) {
    .kick, #overlay h1, #overlay p { opacity: 1; transform: none; }
  }
  @keyframes rise { to { opacity: 1; transform: translateY(0); } }
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

    // Wireframe icosahedron (cyan signal)
    const geo = new THREE.IcosahedronGeometry(1.3, 1);
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x2ee6ff, transparent: true, opacity: 0.85 })
    );
    group.add(wire);

    // Two point-cloud shells -> volumetric "scan" with a faint halo
    function shell(count, radius, color, size, opacity) {
      const pts = [];
      for (let i = 0; i < count; i++) {
        const u = Math.random(), v = Math.random();
        const theta = 2 * Math.PI * u, phi = Math.acos(2 * v - 1);
        pts.push(radius * Math.sin(phi) * Math.cos(theta),
                 radius * Math.sin(phi) * Math.sin(theta),
                 radius * Math.cos(phi));
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      return new THREE.Points(g, new THREE.PointsMaterial({
        color: color, size: size, transparent: true, opacity: opacity
      }));
    }
    group.add(shell(1100, 1.55, 0xbfefff, 0.028, 0.95));  // bright shell
    group.add(shell(700, 1.95, 0x2ee6ff, 0.02, 0.35));    // faint outer halo

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
components.html(HERO_HTML, height=400)


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
    # dark -> cyan colorscale to match the theme
    cyan_scale = [[0.0, "#0a1620"], [0.5, "#0bb8d6"], [1.0, "#2ee6ff"]]
    fig = go.Figure(data=[go.Surface(z=z, colorscale=cyan_scale, showscale=False)])
    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8aa0b6"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Intensity", backgroundcolor="rgba(0,0,0,0)",
                       gridcolor="rgba(120,160,200,0.15)"),
            aspectratio=dict(x=1, y=1, z=0.45),
        ),
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

                # Grad-CAM is opt-in: the gradient pass is the heaviest per-image
                # step, so don't run it on every analyze.
                if st.checkbox("Show Grad-CAM explanation (regions driving the prediction)"):
                    with st.spinner("Computing Grad-CAM..."):
                        try:
                            viz = MedicalVisualization(class_names)
                            fig_cam, _ = viz.create_class_activation_map(model.model, proc, pred_class)
                            # blend the matplotlib figure into the dark theme
                            fig_cam.patch.set_facecolor("#0e141f")
                            for ax in fig_cam.axes:
                                ax.set_facecolor("#0e141f")
                                ax.title.set_color("#e7eef6")
                            st.pyplot(fig_cam, transparent=True)
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
<div style="text-align: center; color: #8aa0b6; padding: 1rem; font-size: 0.85rem;">
    <p>Lung Cancer Detection AI &nbsp;·&nbsp; Powered by TensorFlow &amp; Streamlit</p>
</div>
""", unsafe_allow_html=True)
