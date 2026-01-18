import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pandas as pd
import pickle 

#load the trained models 
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("model.h5")
    with open("label_encoder_gender.pkl","rb") as f:
        label_encoder_gender = pickle.load(f)
    with open("onehot_encoder_geo.pkl","rb") as f:
        onehot_encoder_geo = pickle.load(f)
    with open("scaler.pkl","rb") as f:
        scaler = pickle.load(f)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_all()


## Loading the Streamlit app

st.title(" Customer Churn Prediction ")

#user input

geography = st.selectbox('Georgraphy' , onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_ )
age = st.slider('Age' , 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary') 
tenure = st.slider( 'Tenure' , 0, 10 )
num_of_products = st.slider('Number of Products' , 1,4)
has_cr_card = st.selectbox('Has Credit Card' , [0,1])
is_active_member = st.selectbox( 'Is Active Member ' , [0,1])


# Button to predict
if st.button('Predict Churn'):
    # Encode gender
    gender_encoded = 1 if gender == 'Male' else 0
    
    # Encode geography
    if geography == 'France':
        geo_france, geo_germany, geo_spain = 1, 0, 0
    elif geography == 'Germany':
        geo_france, geo_germany, geo_spain = 0, 1, 0
    else:
        geo_france, geo_germany, geo_spain = 0, 0, 1
    
    # Create input array (Geography repeated 3 times as in training)
    input_array = np.array([[
        credit_score,
        gender_encoded,
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card,
        is_active_member,
        estimated_salary,
        geo_france, geo_germany, geo_spain,
        geo_france, geo_germany, geo_spain,
        geo_france, geo_germany, geo_spain
    ]])
    
    # Scale
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]
    
    st.write(f'Churn Probability: {prediction_proba:.2%}')
    
    if prediction_proba > 0.5:
        st.write('⚠️ The customer is likely to churn.')
    else:
        st.write('✅ The customer is not likely to churn.')