import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime


# --- Thread control for TensorFlow ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main app styling */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Inter', sans-serif;
}

/* Header styling */
.main-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 2rem 0;
    margin-bottom: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    text-align: center;
}

.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.main-subtitle {
    font-size: 1.2rem;
    color: #675a94;
    font-weight: 400;
    margin-bottom: 0;
}

/* Upload section styling */

.upload-header {
    text-align: center;
    margin-bottom: 2rem;
}

.upload-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #3b325c;
    margin-bottom: 0.5rem;
}

.upload-description {
    
    font-size: 1rem;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    width: 100%;
    height: 60px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
}

/* Results styling */
.results-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
            
    border-radius: 20px;
    padding: 0.5rem;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin-bottom: 0.5rem;
    height: 100%;
}

.image-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 0.5rem;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin-bottom: 0.5rem;
    height: 100%;
}

.prediction-card {
    text-align: center;
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
}

.prediction-card.normal {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
    border: 2px solid rgba(34, 197, 94, 0.3);
}

.prediction-card.disease {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
    border: 2px solid rgba(239, 68, 68, 0.3);
}

.prediction-card.warning {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%);
    border: 2px solid rgba(251, 191, 36, 0.3);
}

.prediction-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.prediction-disease {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.prediction-confidence {
    font-size: 1.3rem;
    color: #ffffff;
    font-weight: 500;
}

/* Chart containers */
.chart-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.chart-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
    margin-bottom: 1rem;
}

/* Loading animation */
.loading-container {
    text-align: center;
    padding: 3rem 0;
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border: 4px solid rgba(102, 126, 234, 0.1);
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 1rem auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-text {
    font-size: 1.3rem;
    font-weight: 600;
    
    margin-bottom: 0.5rem;
}

.loading-subtext {
    
    font-size: 1rem;
}

/* Sidebar styling */
.sidebar-content {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Progress bar styling */
.stProgress .st-bo {
    background-color: rgba(102, 126, 234, 0.2);
}

.stProgress .st-bp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Section headers */
.section-header {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    text-align: center;
    margin: 2rem 0 1rem 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
class Config:
    MODEL_PATH = "best_model_B0.keras"  # Keep original model path
    IMAGE_SIZE = (224, 224)
    CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']  # Keep original order and classes
    CLASS_COLORS = ['#ef4444', '#fbbf24', '#3b82f6', '#22c55e']  # Keep original colors
    CLASS_ICONS = ['🔴','🟡','🔵','🟢']  # Keep original icons

# --- Helper Functions ---
@st.cache_resource
def load_eye_model():
    """Load the pre-trained eye disease classification model."""
    try:
        model = load_model(Config.MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure 'model.h5' is in the same directory as this script.")
        return None

def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction.
    
    Args:
        image (PIL.Image): The uploaded image
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for prediction
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image_resized = image.resize(Config.IMAGE_SIZE)
    
    # Convert to array (keeping your original preprocessing)
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    # Note: Not normalizing to keep original preprocessing
    
    return img_array, image_resized

def predict_disease(model, processed_image):
    """
    Make prediction on the processed image.
    
    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    try:
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get class with highest probability
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        predicted_class = Config.CLASS_NAMES[class_index]
        
        return predicted_class, confidence, predictions[0]
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None

def get_confidence_category(confidence):
    """Categorize confidence level for styling."""
    if confidence >= 70:
        return "high"
    elif confidence >= 50:
        return "medium"
    else:
        return "low"

def get_prediction_style(predicted_class):
    """Get styling class based on prediction."""
    if predicted_class == "Normal":
        return "normal"
    elif predicted_class in ["Cataract", "Diabetic Retinopathy", "Glaucoma"]:
        return "disease"
    else:
        return "warning"

# --- Sidebar ---
def create_sidebar():
    """Create and populate the sidebar with instructions and information."""
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload an eye image** using the file uploader
        2. **Click 'Analyze Image'** to start the analysis
        3. **Review the results** including confidence scores
        4. **Consult a healthcare professional** for medical advice
        """)
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About This Tool")
        st.markdown("""
        **Eye Disease Classifier** uses advanced AI to detect common eye conditions:
        
        - **Cataract**: Clouding of the eye's lens
        - **Diabetic Retinopathy**: Diabetes-related eye damage
        - **Glaucoma**: Optic nerve damage condition
        - **Normal**: Healthy eye condition
        
        **⚠️ Important:** This tool is for educational and screening purposes only. 
        Always consult with a qualified ophthalmologist for proper diagnosis and treatment.
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔧 Technical Info")
        st.info(f"""
        **Model Architecture:** Deep Learning CNN
        **Image Size:** {Config.IMAGE_SIZE[0]}x{Config.IMAGE_SIZE[1]} pixels
        **Classes:** {len(Config.CLASS_NAMES)} (Cataract, Diabetic Retinopathy, Glaucoma, Normal)
        **Last Updated:** {datetime.now().strftime("%B %Y")}
        """)
        
        st.markdown("---")
        
        st.markdown("### 📞 Support")
        st.markdown("""
        For technical support or questions about this tool, 
        please contact your system administrator.
        """)

# --- Main App ---
def main():
    """Main application function."""
    
    # Create sidebar
    create_sidebar()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title"> Eye Disease Classifier</h1>
        <p class="main-subtitle">Professional AI-powered eye disease detection for preliminary screening</p>


    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_eye_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check the model file.")
        st.stop()
    
    # Upload section
    st.markdown("""
    <div class="upload-container">
        <div class="upload-header">
            <h2 class="upload-title">📤  Upload Eye Image</h2>
            <p class="upload-description">Upload a clear, high-quality image of an eye for analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an eye image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an eye (JPG, JPEG, or PNG format)"
    )
    
    if uploaded_file is not None:
        # Load and display image
        try:
            image = Image.open(uploaded_file)
            processed_image, resized_image = preprocess_image(image)
                        
            # Analyze button
            if st.button("🔍 **ANALYZE IMAGE**", type="primary"):
                analyze_image(model, processed_image, resized_image, uploaded_file.name)
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.stop()

def analyze_image(model, processed_image, display_image, filename):
    """Perform image analysis and display results."""
    
    # Loading animation
    progress_text = "🧠 AI Analysis in Progress..."
    progress_bar = st.progress(0, text=progress_text)

    # Update progress bar gradually
    for i in range(100):
        time.sleep(0.02)  # Adjust speed for smoothness
        progress_bar.progress(i + 1, text=progress_text)
    
    # Make prediction
    predicted_class, confidence, all_probabilities = predict_disease(model, processed_image)
    
    if predicted_class is None:
        return
    
    # Results section header
    st.markdown('<h2 class="section-header">📊 Analysis Results</h2> ', unsafe_allow_html=True)
    
    # Main Results Section - Image and Prediction Side by Side
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Left Column - Image
    with col1:
        st.markdown("""
        <div class="image-container">
            <h3 style="text-align: center; color: #1e293b; margin-bottom: 1rem;"> 👁️ Analyzed Image</h3>
        </div>
        """, unsafe_allow_html=True)
        # Remove the extra container div and display image directly
        st.image(display_image, caption=f"Uploaded Image: {filename}", use_container_width=True)
    
    # Right Column - Main Prediction and Quick Stats
    with col2:
        st.markdown("""
        <div class="results-container">
            <h3 style="text-align: center; color: #1e293b; margin-bottom: 1rem;"> 🧠 Classification Result</h3>
        """, unsafe_allow_html=True)
        
        # Main prediction card
        prediction_style = get_prediction_style(predicted_class)
        class_index = Config.CLASS_NAMES.index(predicted_class)
        
        st.markdown(f"""
            <div class="prediction-card {prediction_style}">
                <div class="prediction-icon">{Config.CLASS_ICONS[class_index]}</div>
                <div class="prediction-disease">{predicted_class}</div>
                <div class="prediction-confidence">Confidence: {confidence:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Quick stats in right column
        st.markdown("#### 📈 Quick Statistics")
        for i, (class_name, probability) in enumerate(zip(Config.CLASS_NAMES, all_probabilities)):
            confidence_pct = probability * 100
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"**{Config.CLASS_ICONS[i]} {class_name}**")
            
            with col2:
                # Convert numpy float32 to Python float for progress bar
                st.progress(float(probability))
            
            with col3:
                st.markdown(f"**{confidence_pct:.1f}%**")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    

    # Recommendations Section Based on Disease
    st.markdown('<h2 class="section-header">💡 Recommendations & Next Steps</h2>', unsafe_allow_html=True)
    
    get_disease_recommendations(predicted_class, confidence)
    
    # Medical disclaimer
    st.warning("""
    ⚠️ **Medical Disclaimer:** This AI tool is designed for educational and preliminary screening purposes only. 
    The results should not be used as a substitute for professional medical diagnosis. 
    Please consult with a qualified ophthalmologist or healthcare provider for proper medical advice, 
    diagnosis, and treatment of eye conditions.
    """)

def get_disease_recommendations(predicted_class, confidence):
    """Provide specific recommendations based on the predicted disease and confidence level."""
    
    if predicted_class == "Normal":
    
        if round(confidence) >= 70:
            st.success("""
            ### ✅ **Excellent News - Healthy Eyes Detected!**
            
            **What this means:**
            - Your eye appears to be in good health
            - No signs of the tested conditions were detected
            - The AI model is highly confident in this assessment
            
            **Recommended Actions:**
            - 🏥 Continue regular eye check-ups (annually for adults under 60, every 6 months for 60+)
            - 🥕 Maintain a healthy diet rich in vitamins A, C, and E
            - 🕶️ Protect your eyes from UV rays with quality sunglasses
            - 💻 Take regular breaks from screen time (20-20-20 rule)
            - 🚭 Avoid smoking and limit alcohol consumption
            - 💧 Stay hydrated and get adequate sleep
            """)
        else:
            st.info("""
            ### ℹ️ **Likely Normal - Low Confidence**
            
            **What this means:**
            - The image suggests normal eye condition, but with lower confidence
            - The image quality or angle might affect the analysis
            
            **Recommended Actions:**
            - 📸 Consider retaking the photo with better lighting and focus
            - 🏥 Schedule a routine eye exam to confirm
            - 👀 Monitor for any changes in vision or eye comfort
            """)
    
    elif predicted_class == "Cataract":
        if round(confidence) >= 70:
            st.error("""
            ### 🔴 **Cataract Detected - High Confidence**
            
            **What this means:**
            - Clouding of the eye's natural lens has been detected
            - This is a common condition, especially in older adults
            - Cataracts can cause blurry vision, glare sensitivity, and color changes
            
            **Immediate Actions Required:**
            - 🚨 **Schedule an ophthalmologist appointment within 1-2 weeks**
            - 📋 Prepare a list of vision symptoms you've noticed
            - 💊 Review your medications with your doctor
            
            **What to Expect:**
            - 🔍 Comprehensive eye examination
            - 💡 Discussion of treatment options (surgery is highly effective)
            - 👓 Possible temporary vision aids while planning treatment
            
            **Lifestyle Adjustments:**
            - 💡 Use brighter lighting for reading and close work
            - 🕶️ Wear sunglasses to reduce glare
            - 🚗 Avoid driving at night if vision is impaired
            """)
        else:
            st.warning("""
            ### ⚠️ **Possible Cataract - Moderate Confidence**
            
            **Recommended Actions:**
            - 🏥 Schedule an eye exam within 2-4 weeks
            - 👀 Monitor vision changes
            - 📸 Consider retaking the photo for better analysis
            """)
    
    elif predicted_class == "Diabetic Retinopathy":
        if round(confidence) >= 70:
            st.error("""
            ### 🔴 **Diabetic Retinopathy Detected - High Confidence**
            
            **What this means:**
            - Damage to blood vessels in the retina due to diabetes
            - This is a serious complication that can lead to vision loss
            - Early detection and treatment are crucial
            
            **Immediate Actions Required:**
            - 🚨 **Contact your ophthalmologist and endocrinologist immediately**
            - 📅 Schedule appointments within 1 week if possible
            - 🩸 Check your recent blood sugar and HbA1c levels
            
            **Critical Management Steps:**
            - 🍎 Strict blood sugar control is essential
            - 💊 Review diabetes medications with your doctor
            - 🩺 Monitor blood pressure (hypertension worsens the condition)
            - 🚭 Stop smoking immediately if you smoke
            
            **What to Expect:**
            - 🔍 Dilated eye exam and possibly fluorescein angiography
            - 💉 Possible treatments: laser therapy, injections, or surgery
            - 📊 Regular monitoring every 3-6 months
            
            **Lifestyle Changes:**
            - 🥗 Follow a diabetic-friendly diet strictly
            - 🏃‍♀️ Regular exercise as approved by your doctor
            - 💧 Stay well hydrated
            """)
        else:
            st.warning("""
            ### ⚠️ **Possible Diabetic Retinopathy - Moderate Confidence**
            
            **Recommended Actions:**
            - 🏥 Schedule urgent ophthalmologist appointment (within 1-2 weeks)
            - 🩸 Check blood sugar levels immediately
            - 📋 Review diabetes management with your doctor
            """)
    
    elif predicted_class == "Glaucoma":
        if round(confidence) >= 70:
            st.error("""
            ### 🔴 **Glaucoma Detected - High Confidence**
            
            **What this means:**
            - Damage to the optic nerve, often related to eye pressure
            - "Silent thief of sight" - often no early symptoms
            - Can lead to irreversible vision loss if untreated
            
            **Immediate Actions Required:**
            - 🚨 **Schedule an urgent ophthalmologist appointment within 1 week**
            - 🏥 This may require immediate treatment to prevent vision loss
            - 📊 Prepare family history of eye diseases
            
            **What to Expect:**
            - 📏 Eye pressure measurement (tonometry)
            - 🔍 Optic nerve examination
            - 👀 Visual field testing
            - 📸 OCT scan of the optic nerve
            
            **Treatment Options:**
            - 💧 Eye drops to lower pressure (most common first treatment)
            - 💊 Oral medications if needed
            - ⚡ Laser therapy
            - 🏥 Surgery in advanced cases
            
            **Important Notes:**
            - ⏰ Early treatment can prevent further vision loss
            - 👀 Vision loss from glaucoma cannot be recovered
            - 📅 Regular monitoring is essential for life
            
            **Lifestyle Adjustments:**
            - 🧘‍♀️ Manage stress (can affect eye pressure)
            - 🏃‍♀️ Regular moderate exercise (helps lower eye pressure)
            - ☕ Limit caffeine intake
            - 💧 Stay hydrated but avoid large amounts of fluid at once
            """)
        else:
            st.warning("""
            ### ⚠️ **Possible Glaucoma - Moderate Confidence**
            
            **Recommended Actions:**
            - 🏥 Schedule ophthalmologist appointment within 1-2 weeks
            - 👀 Monitor for vision changes or eye pain
            - 📸 Consider retaking the photo for better analysis
            """)
    
    # General advice for all conditions
    st.info("""
    ### 📞 **Emergency Signs - Seek Immediate Medical Attention:**
    - Sudden vision loss or significant vision changes
    - Severe eye pain
    - Flashing lights or new floaters
    - Curtain or shadow in your field of vision
    - Halos around lights with eye pain
    
    ### 🏥 **How to Find an Eye Care Professional:**
    - Ask your primary care doctor for a referral
    - Contact your insurance provider for covered specialists
    - Use online directories from professional organizations
    - Call local hospitals for ophthalmology departments
    """)

if __name__ == "__main__":
    main()