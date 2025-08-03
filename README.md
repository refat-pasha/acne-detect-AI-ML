Acne Detection AI/ML
A machine learning project for classifying skin images as acne or healthy using deep learning models. This tool aims to enable automated and rapid evaluation of facial images for acne detection, leveraging popular architectures and modern Python frameworks.

🌐 Demo
Try the live Streamlit demo:
acnedetectionlite.streamlit.app

🚀 Features
Upload facial images and get instant predictions: "Acne" or "Healthy"

Supports models like VGG19, ResNet50 (see ResNet50.keras)

Includes pre-trained TensorFlow Lite and Keras models (model.tflite, .keras)

Scripts for model training, conversion, and prediction

Modern Python stack (TensorFlow/Keras, Streamlit)

Simple, open, and ready for further improvement

📁 Repository Structure
text
├── main.py                # Primary training or prediction script
├── main2.py               # Additional script for experimentation
├── requirements.txt       # Python dependencies
├── ResNet50.keras         # Saved Keras model
├── model.tflite           # TensorFlow Lite model for mobile/edge usage
├── modelConvert.ipynb     # Notebook for model conversion/export
├── .gitattributes         # Git LFS tracking for large files
├── README.md              # You are here!
🏁 Quickstart
1. Clone this repository

bash
git clone https://github.com/refat-pasha/acne-detect-AI-ML.git
cd acne-detect-AI-ML
2. Install dependencies

bash
pip install -r requirements.txt
3. Run Locally

bash
streamlit run main.py
(or start from the relevant script as needed)

⚙️ Model Information
Transfer learning with state-of-the-art architectures (ResNet50, VGG19, etc.)

Model files tracked via Git LFS to handle large size limits

Includes both Keras (.keras) and TFLite (.tflite) formats

🧑💻 Contributing
Contributions, suggestions, and pull requests are welcome!

Report issues via the "Issues" tab

Submit improvements via Pull Requests

For large dataset/model contribution, please use Git LFS

⚠️ Disclaimer
This project is for educational, research, and demonstration purposes only.
Not intended for clinical or diagnostic use.
For any health concerns, consult a medical professional.

📄 License
MIT License (or specify your license)
