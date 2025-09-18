import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑⚕️")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Multiple Disease Prediction System</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .success-box {
        background-color: #06290e;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #28a745;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open('./sav files/heart_disease_model.sav', 'rb') as f:
            heart_model = pickle.load(f)
        with open('./sav files/parkinsons_model.sav', 'rb') as f:
            parkinsons_model = pickle.load(f)
        with open('./sav files/breast_cancer.sav', 'rb') as f:
            breast_model = pickle.load(f)
        with open('./sav files/diabetes_model.sav', 'rb') as f:
            diabetes_model = pickle.load(f)
        return heart_model, parkinsons_model, breast_model, diabetes_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

heart_disease_model, parkinsons_model, breast_cancer_model, diabetes_model = load_models()

with st.sidebar:
    selected = option_menu(
        'Select disease',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'file-medical'],
        default_index=0
    )

if selected == 'Breast Cancer Prediction':
    st.title('👩⚕️ Breast Cancer Prediction using ML')
    
    if breast_cancer_model is None:
        st.error("Model not loaded.")
    else:
        features = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        sample_vals = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                      1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                      25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
        
        if st.button('Fill Sample Values'):
            for i, val in enumerate(sample_vals):
                st.session_state[f'bc_{i}'] = str(val)
        
        inputs = {}
        cols = st.columns(5)
        for i, feature in enumerate(features):
            with cols[i % 5]:
                inputs[feature] = st.text_input(
                    feature.replace('_', ' ').title(), 
                    value=st.session_state.get(f'bc_{i}', ''),
                    key=f'bc_{i}'
                )

        if st.button('Predict'):
            vals = list(inputs.values())
            if all(v.strip() for v in vals):
                try:
                    user_input = np.array([float(v) for v in vals]).reshape(1, -1)
                    pred = breast_cancer_model.predict(user_input)
                    
                    if hasattr(pred, 'shape') and len(pred.shape) > 1:
                        result = float(pred[0][0]) > 0.5
                    else:
                        result = int(pred[0]) == 1
                    
                    if result:
                        st.markdown('<div class="success-box">⚠️ May have Breast Cancer</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ May not have Breast Cancer</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Fill all fields")

elif selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction')
    if diabetes_model:
        if st.button('Fill Sample Values', key='diabetes_sample'):
            sample_vals = ['2', '120', '80', '20', '85', '26.5', '0.5', '45']
            for i, val in enumerate(sample_vals):
                st.session_state[f'diab_{i}'] = val
        
        col1, col2, col3 = st.columns(3)
        with col1:
            preg = st.text_input('Pregnancies', value=st.session_state.get('diab_0', ''), key='diab_0')
            skin = st.text_input('Skin Thickness', value=st.session_state.get('diab_3', ''), key='diab_3')
        with col2:
            glucose = st.text_input('Glucose', value=st.session_state.get('diab_1', ''), key='diab_1')
            insulin = st.text_input('Insulin', value=st.session_state.get('diab_4', ''), key='diab_4')
        with col3:
            bp = st.text_input('Blood Pressure', value=st.session_state.get('diab_2', ''), key='diab_2')
            bmi = st.text_input('BMI', value=st.session_state.get('diab_5', ''), key='diab_5')
        with col1:
            dpf = st.text_input('Diabetes Pedigree', value=st.session_state.get('diab_6', ''), key='diab_6')
        with col2:
            age = st.text_input('Age', value=st.session_state.get('diab_7', ''), key='diab_7')

        if st.button('Predict Diabetes'):
            vals = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
            if all(v.strip() for v in vals):
                try:
                    user_input = np.array([float(v) for v in vals]).reshape(1, -1)
                    pred = diabetes_model.predict(user_input)
                    if pred[0] == 1:
                        st.markdown('<div class="success-box">⚠️ May be Diabetic</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ May not be Diabetic</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Fill all fields")

elif selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction')
    if heart_disease_model:
        if st.button('Fill Sample Values', key='heart_sample'):
            sample_vals = ['54', '1', '2', '130', '250', '0', '1', '160', '0', '1.2', '1', '0', '2']
            for i, val in enumerate(sample_vals):
                st.session_state[f'heart_{i}'] = val
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input('Age', value=st.session_state.get('heart_0', ''), key='heart_0')
            trestbps = st.text_input('Resting BP', value=st.session_state.get('heart_3', ''), key='heart_3')
            restecg = st.text_input('Resting ECG', value=st.session_state.get('heart_6', ''), key='heart_6')
            oldpeak = st.text_input('ST Depression', value=st.session_state.get('heart_9', ''), key='heart_9')
        with col2:
            sex = st.text_input('Sex (1=M, 0=F)', value=st.session_state.get('heart_1', ''), key='heart_1')
            chol = st.text_input('Cholesterol', value=st.session_state.get('heart_4', ''), key='heart_4')
            thalach = st.text_input('Max Heart Rate', value=st.session_state.get('heart_7', ''), key='heart_7')
            slope = st.text_input('ST Slope', value=st.session_state.get('heart_10', ''), key='heart_10')
        with col3:
            cp = st.text_input('Chest Pain Type', value=st.session_state.get('heart_2', ''), key='heart_2')
            fbs = st.text_input('Fasting Blood Sugar', value=st.session_state.get('heart_5', ''), key='heart_5')
            exang = st.text_input('Exercise Angina', value=st.session_state.get('heart_8', ''), key='heart_8')
            ca = st.text_input('Major Vessels', value=st.session_state.get('heart_11', ''), key='heart_11')
        with col1:
            thal = st.text_input('Thal', value=st.session_state.get('heart_12', ''), key='heart_12')

        if st.button('Predict Heart Disease'):
            vals = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            if all(v.strip() for v in vals):
                try:
                    user_input = np.array([float(v) for v in vals]).reshape(1, -1)
                    pred = heart_disease_model.predict(user_input)
                    if pred[0] == 1:
                        st.markdown('<div class="success-box">❤️ May have Heart Disease</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">💚 May not have Heart Disease</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Fill all fields")

elif selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    if parkinsons_model:
        features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                   'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                   'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                   'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        
        sample_vals = ['119.992', '157.302', '74.997', '0.00784', '0.00007', '0.0037', '0.00554',
                      '0.01109', '0.04374', '0.426', '0.02182', '0.0313', '0.02971', '0.06545',
                      '0.02211', '21.033', '0.414783', '0.815285', '-4.813031', '0.266482',
                      '2.301442', '0.284654']
        
        if st.button('Fill Sample Values', key='parkinsons_sample'):
            for i, val in enumerate(sample_vals):
                st.session_state[f'park_{i}'] = val
        
        inputs = {}
        cols = st.columns(5)
        for i, feature in enumerate(features):
            with cols[i % 5]:
                inputs[feature] = st.text_input(feature, value=st.session_state.get(f'park_{i}', ''), key=f'park_{i}')

        if st.button("Predict Parkinson's"):
            vals = list(inputs.values())
            if all(v.strip() for v in vals):
                try:
                    user_input = np.array([float(v) for v in vals]).reshape(1, -1)
                    pred = parkinsons_model.predict(user_input)
                    if pred[0] == 1:
                        st.markdown('<div class="success-box">🧠 May have Parkinson\'s</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">🧠 May not have Parkinson\'s</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Fill all fields")