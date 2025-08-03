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
    from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
    from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
    from tensorflow.keras.preprocessing.image import img_to_array
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
    .result-danger  { background:#f8d7da;color:#721c24;padding:1rem;border-radius:5px; }
    .info-box       { background:#e7f3ff;color:#0c5460;padding:1rem;border-radius:5px;border-left:4px solid #b8daff;margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING for TFLite --------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tflite_model(model_path="model.tflite"):
    if not os.path.exists(model_path):
        st.error(f"❌ '{model_path}' not found!")
        return None
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    st.success(f"✅ TFLite model loaded: '{model_path}'")
    return interpreter

# --- IMAGE PRE-PROCESSING for TFLite -------------------------------------
def preprocess_image(img: Image.Image, input_shape) -> np.ndarray:
    # Accepts NHWC (batch, height, width, channels)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Normally for classic CNNs, use (224, 224)
    target_size = (input_shape[1], input_shape[2])
    arr = img.resize(target_size)
    arr = img_to_array(arr)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    # Standard scaling for VGG/Xception, use preprocess_input for tflite created from such
    arr = vgg19_preprocess(arr)
    return arr

# --- PREDICTION using TFLite ---------------------------------------------
def predict_acne_tflite(interpreter, arr: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    if preds.size == 1:  # Binary
        idx = 1 if preds[0][0] > 0.5 else 0
        conf = float(preds[0][0] if idx == 1 else 1 - preds[0][0])
    else:  # Multi-class
        probs = preds[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
    return idx, conf, preds[0]

# --- RESULTS DISPLAY ---------------------------------------------------
def display_results(idx, conf, probs, class_names):
    if idx is None or conf is None:
        st.error("❌ Unable to display results")
        return

    st.markdown("## 🔍 Prediction Results")
    label = class_names[idx] if 0 <= idx < len(class_names) else "Unknown"

    # Display logic
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

    interpreter = load_tflite_model()
    if interpreter is None:
        st.stop()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # e.g., (1, 224, 224, 3)
    st.sidebar.header("📋 How to Use")
    st.sidebar.markdown("""
    1. **Upload** a clear image of skin (JPG/PNG)  
    2. **Wait** for the AI analysis  
    3. **Review** the results carefully  
    4. **Consult** a dermatologist for medical advice
    """)
    st.sidebar.header("ℹ️ Model Information")
    st.sidebar.write(f"**Input shape:** {tuple(input_shape)}")
    st.sidebar.write(f"**Output shape:** {output_details[0]['shape']}")
    st.sidebar.write("**Classes:** Acne, Healthy")
    st.sidebar.write("**Model type:** TensorFlow Lite")

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

            arr = preprocess_image(img, input_shape)
            idx, conf, probs = predict_acne_tflite(interpreter, arr)
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
