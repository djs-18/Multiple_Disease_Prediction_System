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

# Load models with proper error handling
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

def validate_inputs(inputs):
    """Validate that all inputs are numeric and not empty"""
    try:
        return [float(x) for x in inputs if x.strip()]
    except ValueError:
        return None

with st.sidebar:
    selected = option_menu(
        'Select disease',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'file-medical'],
        default_index=0
    )

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title('👩⚕️ Breast Cancer Prediction using ML')
    
    if breast_cancer_model is None:
        st.error("Breast cancer model not loaded. Please check the model file.")
    else:
        # Standard breast cancer dataset features (30 features)
        feature_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        
        # Sample values for demonstration
        sample_values = [
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
        
        st.info("Enter values for all 30 features. Click 'Use Sample Values' to fill with example data.")
        
        if st.button('Use Sample Values'):
            st.session_state.update({f'bc_{i}': str(val) for i, val in enumerate(sample_values)})
        
        inputs = {}
        cols = st.columns(5)
        for i, feature in enumerate(feature_names):
            with cols[i % 5]:
                key = f'bc_{i}'
                default_val = st.session_state.get(key, '')
                inputs[feature] = st.text_input(
                    feature.replace('_', ' ').title(), 
                    value=default_val,
                    placeholder=f'e.g. {sample_values[i]}',
                    key=key
                )

        if st.button('Breast Cancer Test Result'):
            input_values = list(inputs.values())
            
            # Validate inputs
            if not all(val.strip() for val in input_values):
                st.error("Please fill in all 30 feature values.")
            else:
                try:
                    # Convert to float and create numpy array
                    numeric_inputs = [float(val) for val in input_values]
                    user_input = np.array(numeric_inputs).reshape(1, -1)
                    
                    # Make prediction
                    prediction = breast_cancer_model.predict(user_input)
                    
                    # Handle different model output formats
                    if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
                        # Neural network output (probability)
                        prob = float(prediction[0][0]) if prediction.shape[1] == 1 else float(prediction[0][1])
                        result = prob > 0.5
                    else:
                        # Traditional ML model output
                        result = int(prediction[0]) == 1
                    
                    if result:
                        st.markdown('<div class="success-box">⚠️ The person <b>may have Breast Cancer</b><br>Please consult a healthcare professional</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ The person <b>may not have Breast Cancer</b><br>Regular checkups are still recommended</div>', unsafe_allow_html=True)
                        
                except ValueError as e:
                    st.error(f"Invalid input values. Please enter numeric values only. Error: {e}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# Other prediction pages (simplified for brevity)
elif selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using ML')
    if diabetes_model is None:
        st.error("Diabetes model not loaded.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.text_input('Pregnancies', placeholder='e.g. 2')
            skin_thickness = st.text_input('Skin Thickness', placeholder='e.g. 20')
            dpf = st.text_input('Diabetes Pedigree Function', placeholder='e.g. 0.5')
        with col2:
            glucose = st.text_input('Glucose Level', placeholder='e.g. 120')
            insulin = st.text_input('Insulin Level', placeholder='e.g. 85')
            age = st.text_input('Age', placeholder='e.g. 45')
        with col3:
            bp = st.text_input('Blood Pressure', placeholder='e.g. 80')
            bmi = st.text_input('BMI', placeholder='e.g. 26.5')

        if st.button('Diabetes Test Result'):
            inputs = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
            if all(val.strip() for val in inputs):
                try:
                    user_input = np.array([float(val) for val in inputs]).reshape(1, -1)
                    prediction = diabetes_model.predict(user_input)
                    if prediction[0] == 1:
                        st.markdown('<div class="success-box">⚠️ The person <b>may be diabetic</b></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ The person <b>may not be diabetic</b></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please fill in all fields.")

elif selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')
    st.info("Heart disease prediction requires 13 parameters. Please fill all fields.")

elif selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")
    st.info("Parkinson's prediction requires 22 voice measurement parameters.")