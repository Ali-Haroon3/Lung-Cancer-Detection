"""Load a committed, pre-trained model artifact produced by train.py.

Kept free of Streamlit so it can be used from any context. The Streamlit
session glue lives in utils/model_state.py.
"""
import json
import os

_DIR = os.path.dirname(__file__)
ARTIFACT_PATH = os.path.join(_DIR, "lung_cancer_model.keras")
LABELS_PATH = os.path.join(_DIR, "labels.json")

# If the artifact isn't committed alongside the code (e.g. a code-only Hugging
# Face Space), download it once from this URL. Override with the MODEL_URL env var.
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://raw.githubusercontent.com/Ali-Haroon3/Lung-Cancer-Detection/main/models/lung_cancer_model.keras",
)

_DEFAULT_META = {
    "class_names": ["normal", "cancer"],
    "architecture": "efficientnetb0",
    "input_shape": [224, 224, 3],
}


def artifact_exists() -> bool:
    """Cheap check (no TensorFlow import) for whether the model is on disk."""
    return os.path.exists(ARTIFACT_PATH)


def model_available() -> bool:
    """True if a model is on disk or can be downloaded — without importing TF."""
    return artifact_exists() or bool(MODEL_URL)


def _download_artifact() -> bool:
    """Download the model to ARTIFACT_PATH. Returns True on success."""
    import urllib.request
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = ARTIFACT_PATH + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, ARTIFACT_PATH)  # atomic: never leave a half file
        return True
    except Exception as e:  # network/URL problems -> fall back to diagnostics
        print(f"Model download failed from {MODEL_URL}: {e}")
        return False


def load_trained_model():
    """Return (LungCancerCNN wrapper, class_names) or (None, None) if unavailable.

    Imports TensorFlow lazily so just checking availability stays cheap, and
    downloads the artifact on demand when it isn't committed with the code.
    """
    if not artifact_exists():
        if not (MODEL_URL and _download_artifact()):
            return None, None

    import tensorflow as tf
    from models.cnn_model import LungCancerCNN

    meta = dict(_DEFAULT_META)
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH) as f:
                meta.update(json.load(f))
        except (ValueError, OSError):
            pass  # fall back to defaults

    keras_model = tf.keras.models.load_model(ARTIFACT_PATH)

    wrapper = LungCancerCNN(
        input_shape=tuple(meta["input_shape"]),
        num_classes=len(meta["class_names"]),
        architecture=meta["architecture"],
    )
    wrapper.model = keras_model
    return wrapper, meta["class_names"]
