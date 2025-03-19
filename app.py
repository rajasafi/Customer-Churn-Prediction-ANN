import streamlit as st
import numpy as np
import tensorflow  as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load the trained model
model = tf.keras.models.load_model('model.h5')

## load the pickle file
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

##streamlit app
st.title('Customer Churn Prediction')

#userInput
geography = st.selectbox('Geography' , label_encoder_geo.categories_[0])
gender = st.selectbox('Gender' , label_encoder_gender.classes_)
age = st.slider('Age' , 18 , 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0 , 10)
num_of_products = st.slider('Number of Products', 1 ,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_number = st.selectbox('Is Active Number',[0,1])

#Example input data
input_data = {
    'CreditScore':[credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_number],
    'EstimatedSalary':[estimated_salary]
}

## Convert to DataFrame
input_df = pd.DataFrame(input_data)

# One-hot encode geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate with the main DataFrame
input_data_combined = pd.concat([input_df, geo_encoded_df], axis=1)

# Scale the data
input_data_scaled = scaler.transform(input_data_combined)

prediction=model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('the customer is likely to churn.')
else:
    st.write('the customer is not likely to churn.')