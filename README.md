# 🏥 Multiple Disease Prediction System using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![ML Models](https://img.shields.io/badge/ML%20Models-Trained-green.svg)](#)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)](#)

A machine learning-powered web application built with Streamlit to predict the presence of common diseases including Diabetes, Heart Disease, Parkinson’s, and Breast Cancer. Users can enter basic health metrics and receive immediate predictions.

---

## 🎯 Project Overview

This project aims to provide a simple yet effective tool for early disease prediction. It features:

- ✅ Trained ML models saved as `.sav` files
- 🎛️ Interactive web interface using Streamlit
- 📦 Easy-to-install Python environment
- 🔍 Real-time disease prediction

---

## 🧠 Supported Diseases

The following diseases are currently supported:

- 🔷 **Diabetes**
- 🔴 **Heart Disease**
- 🟠 **Parkinson’s Disease**
- 🟢 **Breast Cancer**

---

## 🚀 Quick Start

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Multiple_Disease_Prediction_Model.git
cd Multiple_Disease_Prediction_Model

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate         # On Windows
# source venv/bin/activate     # On macOS/Linux

# Install required packages
pip install -r requirements.txt
```

### 🚀 Running the Application

```bash
streamlit run app_final.py
```

The app will open at **http://localhost:8501**


## 📁 Project Structure

```
Multiple_Disease_Prediction_Model/
├── app_final.py                 # Main Streamlit web app
├── requirements.txt             # Python dependencies
├── sav files/                   # Trained model files (.sav)
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   ├── parkinsons_model.sav
│   └── breast_cancer.sav
├── dataset/                     # Training datasets
├── models/                      # Jupyter notebooks
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```


## 🔮 Future Enhancements

### Planned Features
- [ ] Add model accuracy indicators
- [ ] Add model retraining support from UI
- [ ] Integrate user authentication
- [ ] Integration with EEG devices
- [ ] Deploy on Streamlit Cloud or HuggingFace Spaces


---

**⚠️ Medical Disclaimer**: This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

**📅 Last Updated**: June 2025 | **🔢 Version**: 1.0.0
