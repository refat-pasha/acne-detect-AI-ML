import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import warnings


# ── ENV FLAGS ──
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
    from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.utils import register_keras_serializable, get_custom_objects
except Exception as e:
    st.error(f"❌ TensorFlow not installed: {e}")
    st.stop()


st.set_page_config(
    page_title="Acne Skin Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES ----------------------------------------------------
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .result-success { background:#d4edda;color:#155724;padding:1rem;border-radius:5px; }
    .result-warning { background:#fff3cd;color:#856404;padding:1rem;border-radius:5px;border:1px solid #ffeaa7; }
    .result-danger  { background:#f8d7da;color:#721c24;padding:1rem;border-radius:d #f1b0b7; }
    .info-box       { background:#e7f3ff;color:#0c5460;padding:1rem;border-radius:5px;border-left:4px solid #b8daff;margin:1rem 0; }
    .debug-box      { background:#f8f9fa;color:#495057;padding:1rem;border-radius:5px;border:1px solid #dee2e6;margin:1rem 0;font-family:monospace; }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING --------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vgg19_model():
    """Try to load a model file from several common filenames."""
    get_custom_objects().clear()

    @register_keras_serializable()
    def my_lambda_function(x):
        return x

    candidates = [
        #"model.keras", "model.h5",
        #"vgg19_model.keras", "vgg19_model.h5",
        #"Xception.keras", "VGG19.keras", 
        "ResNet50.keras",
        "bestmodel.keras"
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                with st.spinner(f"Loading model from {path}..."):
                    model = keras_load_model(path, compile=False)
                st.success(f"✅ Successfully loaded model from '{path}'")
                st.info(f"📋 Model input shape: {model.input_shape}")
                st.info(f"📋 Model output shape: {model.output_shape}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Failed to load '{path}': {str(e)}")
                continue

    st.error(f"❌ No valid model file found. Tried: {', '.join(candidates)}")
    return None

# --- IMAGE PRE-PROCESSING -------------------------------------
def preprocess_image(img: Image.Image, model_input_shape) -> np.ndarray:
    """Resize & preprocess image to match model expectations."""
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")

        if len(model_input_shape) == 4:              # CNN input
            if model_input_shape[1:3] == (299, 299): # Xception
                target_size = (299, 299)
                arr = img.resize(target_size)
                arr = img_to_array(arr)
                arr = np.expand_dims(arr, axis=0)
                return xception_preprocess(arr)
            else:                                    # VGG19 / ResNet50
                target_size = (224, 224)
                arr = img.resize(target_size)
                arr = img_to_array(arr)
                arr = np.expand_dims(arr, axis=0)
                return vgg19_preprocess(arr)

        # Flattened models
        img = img.convert('L')
        img = img.resize((224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        return arr / 255.0                           # Normalisation
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def reshape_for_flat_model(arr: np.ndarray, input_shape):
    """Adapt image array to flattened network input if required."""
    try:
        if len(input_shape) == 2:                    # Flattened input
            num_features = int(input_shape[1])
            flattened = tf.reshape(arr, (arr.shape[0], -1))
            current_size = flattened.shape[1]

            if current_size != num_features:
                if current_size < num_features:
                    padding = num_features - current_size
                    flattened = tf.pad(flattened, [[0, 0], [0, padding]], 'constant')
                else:
                    flattened = flattened[:, :num_features]
            return flattened.numpy()
        return arr                                   # Already correct
    except Exception as e:
        st.error(f"Error reshaping for flat model: {e}")
        return None

# --- PREDICTION ------------------------------------------------
def predict_acne(model, arr: np.ndarray):
    """Return (predicted_index, confidence, raw_probabilities)."""
    try:
        input_shape = model.input_shape
        inp = arr if len(input_shape) == 4 else reshape_for_flat_model(arr, input_shape)
        if inp is None:
            return None, None, None

        preds = model.predict(inp, verbose=0)

        # Flatten predictions as needed
        preds = preds[0] if preds.ndim == 2 and preds.shape[0] == 1 else preds.flatten()

        if preds.size == 0 or np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            st.error("❌ Invalid predictions returned by model")
            return None, None, None

        if preds.size == 1:                          # Binary model
            idx = 1 if preds[0] > 0.5 else 0
            conf = float(preds[0] if idx == 1 else 1 - preds[0])
        else:                                        # Multi-class
            if not np.isclose(preds.sum(), 1, atol=0.1):
                preds = tf.nn.softmax(preds).numpy()
            idx = int(np.argmax(preds))
            conf = float(preds[idx])

        return idx, conf, preds
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None, None, None

# --- RESULTS DISPLAY (simplified without confidence) -----------
def display_results(idx, conf, probs, class_names):
    if idx is None or conf is None:
        st.error("❌ Unable to display results")
        return

    st.markdown("## 🔍 Prediction Results")

    # Validate index and label
    if 0 <= idx < len(class_names):
        label = class_names[idx]
    else:
        st.warning("⚠️ Prediction index out of range – applying fallback")
        idx = max(0, min(idx, len(class_names) - 1))
        label = class_names[idx]

    # Simple result display without confidence score
    if label.lower() == "acne":
        st.markdown(f"""<div class="result-danger">
            <h3>⚠️ Acne Detected</h3>
            <p>Signs of acne detected in the uploaded image. Consider consulting a dermatologist for proper evaluation and treatment.</p>
        </div>""", unsafe_allow_html=True)
    elif label.lower() == "healthy":
        st.markdown(f"""<div class="result-success">
            <h3>✅ Healthy Skin</h3>
            <p>No signs of acne detected in the uploaded image. Your skin appears to be healthy.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-warning">
            <h3>❓ Result: {label}</h3>
            <p>Classification completed. Please verify with a healthcare professional for accurate diagnosis.</p>
        </div>""", unsafe_allow_html=True)

# --- MISC ------------------------------------------------------
def display_medical_disclaimer():
    st.markdown("""<div class="info-box">
        <h4>⚕️ Medical Disclaimer</h4>
        <p>This AI tool is for educational and informational purposes only. 
                It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist or healthcare provider.</p>
    </div>""", unsafe_allow_html=True)

def validate_image(file) -> Image.Image:
    if file.size > 10 * 1024 * 1024:
        st.error("❌ File too large (maximum 10 MB).")
        return None
    try:
        img = Image.open(file)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
    except Exception as e:
        st.error(f"❌ Invalid image file: {e}")
        return None
    if img.width < 100 or img.height < 100:
        st.warning("⚠️ Image resolution is low; results may be less accurate.")
    return img

# --- MAIN ------------------------------------------------------
def main():
    st.title("🩺 Acne Skin Disease Prediction")
    st.markdown("### AI-Powered Skin Analysis Using Deep Learning")

    model = load_vgg19_model()
    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.header("📋 How to Use")
    st.sidebar.markdown("""
    1. **Upload** a clear image of skin (JPG/PNG)  
    2. **Wait** for the AI analysis  
    3. **Review** the results carefully  
    4. **Consult** a dermatologist for medical advice
    """)

    st.sidebar.header("ℹ️ Model Information")
    st.sidebar.write(f"**Input shape:** {model.input_shape}")
    st.sidebar.write(f"**Output shape:** {model.output_shape}")
    st.sidebar.write("**Classes:** Acne, Healthy")
    st.sidebar.write("**Model type:** Deep Learning CNN")

    class_names = ["Acne", "Healthy"]
    st.info(f"📋 Current class configuration: {class_names}")

    uploaded_file = st.file_uploader(
        "Upload a Skin Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit skin image"
    )

    if uploaded_file is not None:
        img = validate_image(uploaded_file)
        if img is not None:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.write(f"**Size:** {img.width}×{img.height}px  |  **Mode:** {img.mode}")

            arr = preprocess_image(img, model.input_shape)
            if arr is not None:
                idx, conf, probs = predict_acne(model, arr)
                display_results(idx, conf, probs, class_names)
                


                if st.button("📥 Download Results"):
                    result_data = {
                        "Prediction": "Acne Detected" if class_names[idx].lower() == "acne" else "Healthy Skin",
                        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Image_Name": uploaded_file.name
                    }
                    csv = pd.DataFrame([result_data]).to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        data=csv,
                        file_name=f"acne_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("👆 Please upload an image to begin analysis")

    display_medical_disclaimer()
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#000000;'>🤖 Powered by refat | For educational purposes only</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
