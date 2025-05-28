# import streamlit as st
# import os
# import time
# import numpy as np
# import pandas as pd
# import joblib
# import imagehash
# from PIL import Image
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Handle OpenCV import with fallback
# try:
#     import cv2
# except ImportError as e:
#     st.error(f"OpenCV import error: {e}")
#     st.error("Please check the requirements.txt and packages.txt files")
#     st.stop()

# # === Streamlit Configuration ===
# st.set_page_config(
#     page_title="Fraud Image Detection",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .prediction-fraud {
#         background-color: #ffebee;
#         border: 2px solid #f44336;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
#     .prediction-good {
#         background-color: #e8f5e8;
#         border: 2px solid #4caf50;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 1rem 0;
#     }
#     .stApp {
#         max-width: 1200px;
#         margin: 0 auto;
#     }
# </style>
# """, unsafe_allow_html=True)

# # === Load Models and Scalers ===
# @st.cache_resource
# def load_models():
#     """Load all required models and scalers with caching"""
#     try:
#         model_files = {
#             'scaler': 'feature_scaler.pkl',
#             'selector': 'feature_selector.pkl', 
#             'xgb_model': 'xgb_classifier.pkl',
#             'lgb_model': 'lgbm_classifier.pkl'
#         }
        
#         models = {}
#         for name, filename in model_files.items():
#             if os.path.exists(filename):
#                 models[name] = joblib.load(filename)
#             else:
#                 st.error(f"❌ {filename} not found.")
#                 return None
        
#         st.sidebar.success("✅ Models loaded successfully!")
#         return models['scaler'], models['selector'], models['xgb_model'], models['lgb_model']
        
#     except Exception as e:
#         st.error(f"❌ Error loading models: {e}")
#         return None

# @st.cache_data
# def load_feature_extraction():
#     """Load feature extraction function"""
#     try:
#         from Model2 import extract_features, IMG_SIZE
#         return extract_features, IMG_SIZE
#     except ImportError as e:
#         st.error(f"❌ Could not import Model2.py: {e}")
#         return None, None

# # === Helper Functions ===
# def compute_image_hash(image):
#     """Compute perceptual hash of an image"""
#     try:
#         if isinstance(image, np.ndarray):
#             image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         else:
#             image_pil = image.convert('RGB')
#         return imagehash.phash(image_pil)
#     except Exception:
#         return None

# def process_image(uploaded_file, scaler, selector, xgb_model, lgb_model, extract_features, IMG_SIZE, threshold=0.5):
#     """Process uploaded image and make prediction"""
#     try:
#         # Read image from uploaded file
#         image = Image.open(uploaded_file)
        
#         # Convert PIL image to OpenCV format
#         img_array = np.array(image)
#         if len(img_array.shape) == 3:
#             if img_array.shape[2] == 4:  # RGBA
#                 img_array = img_array[:, :, :3]  # Remove alpha channel
#             img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#         elif len(img_array.shape) == 2:  # Grayscale
#             img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
#         else:
#             raise ValueError("Invalid image format")
        
#         # Resize image
#         img = cv2.resize(img, IMG_SIZE)
        
#         # Convert to YCrCb color space
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
#         # Extract features
#         features = extract_features(img)
#         features = np.array(features).reshape(1, -1)
        
#         # Scale and select features
#         features_scaled = scaler.transform(features)
#         features_selected = selector.transform(features_scaled)
        
#         # Make predictions
#         xgb_proba = xgb_model.predict_proba(features_selected)[0, 1]
#         lgb_proba = lgb_model.predict_proba(features_selected)[0, 1]
#         ensemble_proba = (xgb_proba + lgb_proba) / 2
        
#         # Determine label and confidence
#         label = 'Fraud' if ensemble_proba >= threshold else 'Good'
#         confidence = ensemble_proba if ensemble_proba >= 0.5 else 1 - ensemble_proba
        
#         return {
#             'prediction': label,
#             'confidence': confidence,
#             'ensemble_probability': ensemble_proba,
#             'xgb_probability': xgb_proba,
#             'lgb_probability': lgb_proba,
#             'original_image': image
#         }
        
#     except Exception as e:
#         return {'error': str(e)}

# def create_probability_chart(result, threshold):
#     """Create probability visualization chart"""
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     models = ['XGBoost', 'LightGBM', 'Ensemble']
#     probabilities = [
#         result['xgb_probability'],
#         result['lgb_probability'],
#         result['ensemble_probability']
#     ]
    
#     colors = ['#3498db', '#2ecc71', '#e74c3c']
#     bars = ax.bar(models, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
#     # Add threshold line
#     ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
#                label=f'Threshold ({threshold})')
    
#     # Customize chart
#     ax.set_ylabel('Fraud Probability', fontsize=12, fontweight='bold')
#     ax.set_title('Model Predictions Comparison', fontsize=14, fontweight='bold')
#     ax.set_ylim(0, 1)
#     ax.grid(True, alpha=0.3, axis='y')
#     ax.legend()
    
#     # Add value labels on bars
#     for bar, prob in zip(bars, probabilities):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
#                 f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
#     plt.tight_layout()
#     return fig

# # === Main Streamlit App ===
# def main():
#     # Header
#     st.markdown('<h1 class="main-header">🔍 Fraud Image Detection System</h1>', unsafe_allow_html=True)
#     st.markdown("**Upload an image to detect if it's fraudulent or genuine using ensemble ML models**")
    
#     # Load models and feature extraction
#     models = load_models()
#     if not models:
#         st.error("Please ensure all model files are uploaded correctly.")
#         st.stop()
    
#     scaler, selector, xgb_model, lgb_model = models
    
#     extract_features, IMG_SIZE = load_feature_extraction()
#     if not extract_features:
#         st.error("Please ensure Model2.py is uploaded correctly.")
#         st.stop()
    
#     # Sidebar Configuration
#     st.sidebar.header("⚙️ Configuration")
    
#     # Threshold slider
#     threshold = st.sidebar.slider(
#         "Fraud Detection Threshold",
#         min_value=0.1,
#         max_value=0.9,
#         value=0.5,
#         step=0.05,
#         help="Higher threshold = more conservative fraud detection"
#     )
    
#     # Model info
#     st.sidebar.info(f"""
#     **Model Configuration:**
#     - **Models**: XGBoost + LightGBM Ensemble
#     - **Image Size**: {IMG_SIZE}
#     - **Color Space**: YCrCb
#     - **Threshold**: {threshold}
#     """)
    
#     # Main layout
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.subheader("📤 Upload Image")
#         uploaded_file = st.file_uploader(
#             "Select an image file",
#             type=['jpg', 'jpeg', 'png'],
#             help="Supported formats: JPG, JPEG, PNG"
#         )
        
#         if uploaded_file is not None:
#             # Display uploaded image
#             image = Image.open(uploaded_file)
#             st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            
#             # Image details
#             st.write(f"**Image Details:**")
#             st.write(f"- **Size**: {image.size[0]} x {image.size[1]} pixels")
#             st.write(f"- **Mode**: {image.mode}")
#             st.write(f"- **Format**: {uploaded_file.type}")
    
#     with col2:
#         if uploaded_file is not None:
#             st.subheader("🔬 Analysis")
            
#             if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
#                 with st.spinner("🔄 Processing image..."):
#                     start_time = time.time()
#                     result = process_image(
#                         uploaded_file, scaler, selector, xgb_model, lgb_model, 
#                         extract_features, IMG_SIZE, threshold
#                     )
#                     processing_time = time.time() - start_time
                
#                 # Display results
#                 if 'error' in result:
#                     st.error(f"❌ **Error**: {result['error']}")
#                 else:
#                     # Main prediction result
#                     prediction = result['prediction']
#                     confidence = result['confidence']
                    
#                     if prediction == 'Fraud':
#                         st.markdown(f"""
#                         <div class="prediction-fraud">
#                             <h2 style="color: #d32f2f; margin: 0;">🚨 FRAUD DETECTED</h2>
#                             <h3 style="color: #d32f2f; margin: 0;">Confidence: {confidence:.1%}</h3>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     else:
#                         st.markdown(f"""
#                         <div class="prediction-good">
#                             <h2 style="color: #2e7d32; margin: 0;">✅ GENUINE IMAGE</h2>
#                             <h3 style="color: #2e7d32; margin: 0;">Confidence: {confidence:.1%}</h3>
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     # Store results in session state for detailed view
#                     st.session_state.result = result
#                     st.session_state.processing_time = processing_time
    
#     # Detailed Results Section
#     if 'result' in st.session_state:
#         result = st.session_state.result
#         processing_time = st.session_state.processing_time
        
#         st.markdown("---")
#         st.subheader("📊 Detailed Analysis")
        
#         # Metrics row
#         col_a, col_b, col_c, col_d = st.columns(4)
        
#         with col_a:
#             st.metric(
#                 "Ensemble Score", 
#                 f"{result['ensemble_probability']:.3f}",
#                 delta=f"{result['ensemble_probability'] - threshold:.3f}"
#             )
        
#         with col_b:
#             st.metric("XGBoost Score", f"{result['xgb_probability']:.3f}")
        
#         with col_c:
#             st.metric("LightGBM Score", f"{result['lgb_probability']:.3f}")
        
#         with col_d:
#             st.metric("Processing Time", f"{processing_time:.3f}s")
        
#         # Visualization
#         st.subheader("📈 Model Comparison")
#         fig = create_probability_chart(result, threshold)
#         st.pyplot(fig)
        
#         # Detailed breakdown
#         with st.expander("🔍 Detailed Breakdown"):
#             st.json({
#                 'Final_Prediction': result['prediction'],
#                 'Confidence_Level': f"{result['confidence']:.1%}",
#                 'Ensemble_Probability': f"{result['ensemble_probability']:.4f}",
#                 'XGBoost_Probability': f"{result['xgb_probability']:.4f}",
#                 'LightGBM_Probability': f"{result['lgb_probability']:.4f}",
#                 'Detection_Threshold': threshold,
#                 'Processing_Time_Seconds': f"{processing_time:.4f}",
#                 'Image_Dimensions': f"{result['original_image'].size[0]}x{result['original_image'].size[1]}"
#             })
    
#     # Help section
#     with st.expander("ℹ️ How to Use"):
#         st.markdown("""
#         ### 📋 **Quick Start:**
#         1. **Upload** an image using the file uploader
#         2. **Adjust** the detection threshold if needed (sidebar)
#         3. **Click** "Analyze Image" to get predictions
#         4. **Review** detailed results and model comparisons
        
#         ### 🧠 **About the System:**
#         - **Ensemble Model**: Combines XGBoost and LightGBM classifiers
#         - **Feature Engineering**: Advanced feature extraction
#         - **Real-time Analysis**: Instant predictions with confidence scoring
#         - **Threshold Control**: Adjustable sensitivity for different use cases
        
#         ### ⚡ **Performance:**
#         - Typical processing time: 0.05-0.1 seconds per image
#         - Supported formats: JPG, JPEG, PNG
#         - Automatic image preprocessing and resizing
#         """)
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style="text-align: center; color: #666;">
#         🔍 <b>Fraud Detection System</b> | Powered by Machine Learning & Streamlit
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import joblib
from Model2 import extract_features, IMG_SIZE

# Load models and transformers
scaler = joblib.load("feature_scaler.pkl")
selector = joblib.load("feature_selector.pkl")
xgb_model = joblib.load("xgb_classifier.pkl")
lgbm_model = joblib.load("lgbm_classifier.pkl")

# Streamlit UI Setup
st.set_page_config(page_title="Reusable Bag Fraud Detection", layout="centered")

st.markdown("<h1 style='text-align: center; color: green;'>Reusable Bag Image Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload a reusable bag image", type=["jpg", "jpeg", "png"])

def predict(image_np):
    features = extract_features(image_np)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_selected = selector.transform(features_scaled)
    
    # Predict probabilities for the positive class (fraudulent)
    prob1 = xgb_model.predict_proba(features_selected)[0][1]
    prob2 = lgbm_model.predict_proba(features_selected)[0][1]
    
    avg_prob = (prob1 + prob2) / 2  # Average confidence
    final_pred = int(avg_prob >= 0.5)  # Threshold at 0.5
    
    return final_pred, avg_prob

def main():
    if uploaded_file is not None:
        st.image(uploaded_file, caption='🖼️ Uploaded Image', use_container_width=True)
        st.markdown("⌛ **Processing image and extracting features...**")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name

        # Load and preprocess image
        image = cv2.imread(temp_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)

        prediction, confidence = predict(image)

        st.markdown("---")
        if prediction == 1:
            st.error("🔴 **This is a Recaptured (Fraudulent) Image!**")
        else:
            st.success("🟢 **This is an Original Image.**")

        st.markdown(f"**🔍 Confidence:** `{confidence:.2%}`")

        # Clean up temp file
        os.remove(temp_path)

if __name__ == "__main__":
    main()

