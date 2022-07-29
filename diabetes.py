import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib



st.markdown("<h1 style='text-align: center; color: red;'>WELCOME</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: red;'>This app predicts a patient's diabetes status</h2>"
            , unsafe_allow_html=True)

#@st.cache(suppress_st_warning=True)
app_mode = st.sidebar.selectbox('Select Page',['Home','Predict']) #two pages

if app_mode=='Home':
    st.title('DIABETES PREDICTION :')  
    #st.image('loan_image.jpg')
    st.markdown('Dataset')
    data = pd.read_csv('diabetes.csv')
    st.write(data.head(10))
    st.write(data.columns)
    st.markdown('Applicant Income VS Loan Amount ')

    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(data.corr(), annot = True)
    st.pyplot(fig)

elif app_mode == 'Predict':
    
    st.subheader('Fill in the information to know your status')
    
    Age = st.number_input('Enter Age')
    Glucose = st.number_input('Enter Glucose level')
    BloodPressure = st.number_input('Enter Blood Pressure')
    SkinThickness = st.number_input('Enter Skin Thickness')
    Insulin = st.number_input('Enter Insulin')
    BMI = st.number_input('Enter BMI')
    DiabetesPedigreeFunction = st.number_input('Enter Diabetes Pedigree Function')
    st.slider('Age', 0, 100)
    Pregnancies = st.slider('Pregnancies', 0, 10)
    predict_button = st.button('Predict', key = 1)

    data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    single_sample = np.array(data).reshape(1, -1)       
    if predict_button:
        loaded_model = joblib.load('diabetes_model.sav')
        prediction = loaded_model.predict(single_sample)
        if prediction[0] == 0 :
            st.success("You don't have diabetes")
        elif prediction[0] == 1:
            st.error("You may have diabetes")