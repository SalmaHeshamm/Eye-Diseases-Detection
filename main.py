import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import plotly.graph_objects as go
import time

# --- Thread control for TensorFlow ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# --- Page Configuration ---
st.set_page_config(
    page_title="EyeAI",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding: 1rem; max-width: 800px; margin: 0 auto;}
.stApp {background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b69 100%); min-height: 100vh;}
.header {text-align:center; padding:2rem 0; color:white; width:100%; max-width:800px; margin:0 auto;}
.title {font-size:3rem; font-weight:800; background:linear-gradient(45deg,#00f5ff,#ff00ff,#ffff00); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.5rem;}
.subtitle {font-size:1rem; color:rgba(255,255,255,0.8); font-weight:300; line-height:1.6;}
.upload-section {background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border-radius:15px; padding:1.5rem; margin:1rem auto; width:100%; max-width:800px; height:200px; border:1px solid rgba(255,255,255,0.1); display:flex; align-items:center; justify-content:center;}
.upload-area {border:2px dashed rgba(0,245,255,0.5); border-radius:10px; padding:1rem; width:100%; height:100%; text-align:center; background:rgba(0,245,255,0.05); transition: all 0.3s ease;}
.upload-area:hover {border-color:#00f5ff; background:rgba(0,245,255,0.1);}
.upload-text {color:white; font-size:1.2rem; font-weight:600; margin-bottom:0.5rem;}
.analyze-btn {text-align:center; display:flex; justify-content:center; align-items:center; margin-top:1rem;}
.stButton > button {background: linear-gradient(45deg,#00f5ff,#ff00ff); color:white; border:none; border-radius:50px; padding:0.8rem 2rem; font-size:1rem; font-weight:700; text-transform:uppercase; width:300px; height:50px; box-shadow:0 10px 30px rgba(0,245,255,0.4); transition:all 0.3s ease;}
.stButton > button:hover {transform:translateY(-3px); box-shadow:0 15px 40px rgba(0,245,255,0.6); background:linear-gradient(45deg,#ff00ff,#00f5ff);}
.results-section {text-align:center; max-width:800px; margin:2rem auto;}
.prediction-card {background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border-radius:20px; padding:3rem 2rem; text-align:center; margin:2rem 0; border:1px solid rgba(255,255,255,0.1); box-shadow:0 20px 60px rgba(0,0,0,0.3);}
.result-icon {font-size:4rem; margin-bottom:1rem;}
.result-disease {font-size:2.5rem; font-weight:800; color:white; margin-bottom:0.5rem; text-shadow:2px 2px 4px rgba(0,0,0,0.5);}
.result-confidence {font-size:1.3rem; color:rgba(255,255,255,0.8); font-weight:500;}
.high-confidence {background:linear-gradient(135deg,rgba(34,197,94,0.2),rgba(34,197,94,0.1)); border-left:4px solid #22c55e;}
.medium-confidence {background:linear-gradient(135deg,rgba(251,191,36,0.2),rgba(251,191,36,0.1)); border-left:4px solid #fbbf24;}
.low-confidence {background:linear-gradient(135deg,rgba(239,68,68,0.2),rgba(239,68,68,0.1)); border-left:4px solid #ef4444;}
.analysis-grid {display:grid; grid-template-columns:1fr 1fr; gap:2rem; margin:2rem 0;}
.chart-container {background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border-radius:15px; padding:1.5rem; border:1px solid rgba(255,255,255,0.1);}
.chart-title {color:white; font-weight:600; margin-bottom:1rem; text-align:center; font-size:1.2rem;}
.prob-item {background:rgba(255,255,255,0.05); padding:1rem; border-radius:10px; display:flex; justify-content:space-between; align-items:center; border-left:4px solid; transition:all 0.3s ease;}
.prob-name {color:white; font-weight:600; display:flex; align-items:center; gap:0.5rem;}
.prob-value {color:white; font-weight:700; font-size:1.1rem;}
.loading-section {text-align:center; padding:2rem; color:white; width:100%; max-width:800px; height:200px; margin:1rem auto;}
.loading-spinner {width:50px; height:50px; border:4px solid rgba(255,255,255,0.1); border-top:4px solid #00f5ff; border-radius:50%; animation:spin 1s linear infinite; margin:0 auto 1rem auto;}
@keyframes spin {0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
.loading-text {font-size:1.2rem; font-weight:600; margin-bottom:0.5rem; background: linear-gradient(45deg,#00f5ff,#ff00ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.loading-subtext {color:rgba(255,255,255,0.6); font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('''
<div class="header">
    <h1 class="title">👁️ EyeAI </h1>
    <p class="subtitle">Professional AI-powered eye disease detection with instant analysis and high accuracy results</p>
</div>
''', unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_eye_model():
    try:
        return load_model("best_model.keras")
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

model = load_eye_model()
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal' ]
class_colors = ['#ef4444', '#fbbf24', '#3b82f6', '#22c55e']
class_icons = ['🔴','🟡','🟢','🔵']

# --- Upload Section ---
st.markdown('''
<div class="upload-section">
    <div class="upload-area">
        <div class="upload-text">📤 Upload Eye Image</div>
    </div>
</div>
''', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_resized = image.resize((224, 224))
    st.image(img_resized, width=300)
    
    if st.button("🔍 ANALYZE NOW"):
        st.session_state["button_clicked"] = True

    if st.session_state.get("button_clicked", False) and model is not None:
        # Loading
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown('''
            <div class="loading-section">
                <div class="loading-spinner"></div>
                <div class="loading-text">AI Analysis in Progress</div>
                <div class="loading-subtext">Processing neural patterns and identifying conditions...</div>
            </div>
            ''', unsafe_allow_html=True)

        # Progress
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i+1)
            time.sleep(0.02)
        progress_bar.empty()

        # Prediction
        img_array = img_to_array(img_resized) 
        img_array = np.expand_dims(img_array, axis=0) 
        pred = model.predict(img_array)
        print(pred)
        class_index = np.argmax(pred)
        confidence = np.max(pred)*100
        predicted_class = class_names[class_index]

        # Remove loading
        loading_placeholder.empty()

        # Results
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        confidence_class = "high-confidence" if confidence>=85 else "medium-confidence" if confidence>=70 else "low-confidence"
        st.markdown(f'''
        <div class="prediction-card {confidence_class}">
            <div class="result-icon">{class_icons[class_index]}</div>
            <div class="result-disease">{predicted_class}</div>
            <div class="result-confidence">{confidence:.1f}% Confidence</div>
        </div>
        ''', unsafe_allow_html=True)

        # Analysis grid
        st.markdown('<div class="analysis-grid">', unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">📊 Probability Distribution</div></div>', unsafe_allow_html=True)
            fig = go.Figure(data=[go.Bar(x=class_names, y=[float(p)*100 for p in pred[0]], marker_color=class_colors)])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', yaxis=dict(title='Confidence %'))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">🏷️ Detailed Confidence</div></div>', unsafe_allow_html=True)
            for i,name in enumerate(class_names):
                color=class_colors[i]
                st.markdown(f'''
                <div class="prob-item" style="border-left-color:{color}">
                    <div class="prob-name">{class_icons[i]} {name}</div>
                    <div class="prob-value">{pred[0][i]*100:.1f}%</div>
                </div>
                ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # analysis-grid
        st.markdown('</div>', unsafe_allow_html=True)  # results-section
