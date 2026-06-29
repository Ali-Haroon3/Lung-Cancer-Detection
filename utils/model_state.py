"""Streamlit glue for making a committed model artifact available in session.

Loading the Keras model is expensive, so it's cached for the process with
@st.cache_resource; ensure_model_loaded() just copies the cached handle into
the current session_state when no in-session model is present.
"""
import streamlit as st

from models.model_loader import model_available, load_trained_model


@st.cache_resource(show_spinner="Loading trained model (first run may download it)...")
def _get_artifact_model():
    return load_trained_model()  # (wrapper, class_names) or (None, None)


def ensure_model_loaded() -> bool:
    """Ensure a usable model is in session_state. Returns True if one is ready.

    Prefers a model trained in this session; otherwise falls back to the
    committed/downloaded artifact (loaded once, cached for the process).
    """
    if st.session_state.get("model_trained") and st.session_state.get("trained_model") is not None:
        return True

    if model_available():
        model, class_names = _get_artifact_model()
        if model is not None:
            st.session_state.trained_model = model
            st.session_state.model_trained = True
            st.session_state.class_names = class_names
            return True

    return False
