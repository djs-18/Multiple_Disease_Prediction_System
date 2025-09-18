import os
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑⚕️")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Multiple Disease Prediction System</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 350px !important;
    }
    .stTextInput > div > div > input {
        background-color: #f0f0f0;
        color: black;
        border-radius: 8px;
        caret-color: #222 !important;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
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

heart_disease_model = pickle.load(open('./sav files/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('./sav files/parkinsons_model.sav', 'rb'))
breast_cancer_model = pickle.load(open('./sav files/breast_cancer.sav', 'rb'))
diabetes_model = pickle.load(open('./sav files/diabetes_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu(
        'Select disease',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'file-medical'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', placeholder='e.g. 2')
    with col2:
        Glucose = st.text_input('Glucose Level', placeholder='e.g. 120')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', placeholder='e.g. 80')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value', placeholder='e.g. 20')
    with col2:
        Insulin = st.text_input('Insulin Level', placeholder='e.g. 85')
    with col3:
        BMI = st.text_input('BMI value', placeholder='e.g. 26.5')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', placeholder='e.g. 0.5')
    with col2:
        Age = st.text_input('Age of the Person', placeholder='e.g. 45')

    if st.button('Diabetes Test Result'):
        user_input = np.array([float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                              float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]).reshape(1, -1)
        diab_prediction = diabetes_model.predict(user_input)
        if diab_prediction[0] == 1:
            st.markdown('<div class="success-box">⚠️ The person is <b>diabetic</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ The person is <b>not diabetic</b></div>', unsafe_allow_html=True)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', placeholder='e.g. 54')
    with col2:
        sex = st.text_input('Sex (1=Male, 0=Female)', placeholder='e.g. 1')
    with col3:
        cp = st.text_input('Chest Pain Type (0–3)', placeholder='e.g. 2')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure', placeholder='e.g. 130')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', placeholder='e.g. 250')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', placeholder='e.g. 0')
    with col1:
        restecg = st.text_input('Resting ECG Result (0–2)', placeholder='e.g. 1')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved', placeholder='e.g. 160')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)', placeholder='e.g. 0')
    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise', placeholder='e.g. 1.2')
    with col2:
        slope = st.text_input('Slope of the Peak ST Segment (0–2)', placeholder='e.g. 1')
    with col3:
        ca = st.text_input('Major Vessels Colored by Flourosopy (0–3)', placeholder='e.g. 0')
    with col1:
        thal = st.text_input('Thal (0=normal, 1=fixed defect, 2=reversible defect)', placeholder='e.g. 2')

    if st.button('Heart Disease Test Result'):
        user_input = np.array([float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), 
                              float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), 
                              float(ca), float(thal)]).reshape(1, -1)
        heart_prediction = heart_disease_model.predict(user_input)
        if heart_prediction[0] == 1:
            st.markdown('<div class="success-box">❤️ The person <b>has heart disease</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">💚 The person <b>does not have any heart disease</b></div>', unsafe_allow_html=True)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")
    parkinsons_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    inputs = {}
    cols = st.columns(5)
    for i, feature in enumerate(parkinsons_features):
        with cols[i % 5]:
            inputs[feature] = st.text_input(feature, placeholder='e.g. 119.992')

    if st.button("Parkinson's Test Result"):
        user_input = np.array([float(inputs[f]) for f in parkinsons_features]).reshape(1, -1)
        parkinsons_prediction = parkinsons_model.predict(user_input)
        if parkinsons_prediction[0] == 1:
            st.markdown('<div class="success-box">🧠 The person <b>has Parkinson\'s disease</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">🧠 The person <b>does not have Parkinson\'s disease</b></div>', unsafe_allow_html=True)

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title('👩⚕️ Breast Cancer Prediction using ML')
    features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    inputs = {}
    cols = st.columns(5)
    for i, feature in enumerate(features):
        with cols[i % 5]:
            inputs[feature] = st.text_input(feature.replace('_', ' ').title(), placeholder='e.g. 17.99')

    if st.button('Breast Cancer Test Result'):
        user_input = np.array([float(inputs[f]) for f in features]).reshape(1, -1)
        breast_cancer_prediction = breast_cancer_model.predict(user_input)
        if breast_cancer_prediction[0] > 0.5:
            st.markdown('<div class="success-box">🩺 The person <b>has Breast Cancer</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">🩺 The person <b>does not have Breast Cancer</b></div>', unsafe_allow_html=True)