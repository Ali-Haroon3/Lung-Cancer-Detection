"""Load a committed, pre-trained model artifact produced by train.py.

Kept free of Streamlit so it can be used from any context. The Streamlit
session glue lives in utils/model_state.py.
"""
import json
import os

_DIR = os.path.dirname(__file__)
ARTIFACT_PATH = os.path.join(_DIR, "lung_cancer_model.keras")
LABELS_PATH = os.path.join(_DIR, "labels.json")

_DEFAULT_META = {
    "class_names": ["normal", "cancer"],
    "architecture": "efficientnetb0",
    "input_shape": [224, 224, 3],
}


def artifact_exists() -> bool:
    """Cheap check (no TensorFlow import) for whether a trained model is committed."""
    return os.path.exists(ARTIFACT_PATH)


def load_trained_model():
    """Return (LungCancerCNN wrapper, class_names) or (None, None) if no artifact.

    Imports TensorFlow lazily so just checking for an artifact stays cheap.
    """
    if not artifact_exists():
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
