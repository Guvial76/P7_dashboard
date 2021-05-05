import streamlit as st
import pandas as pd
import joblib


st.write("""
# Credit Score App

This app provides information regarding credit decision based on customer scoring

""")

st.sidebar.header('Credit Application Number')

st.sidebar.markdown("""
Please fill the SK_ID_CURR
""")

cust_id = st.sidebar.number_input('SK_ID_CURR',
                                   min_value=100001., 
                                   value=100013., 
                                   step=1.)

data = pd.read_csv('test_data.csv')

df = data.drop('TARGET', axis = 1)

# Read in saved classification model
load_clf = joblib.load("trained_model.joblib")

# Apply model to make predictions
prediction_proba = load_clf.predict_proba(df.loc[df['SK_ID_CURR'] = cust_id])

# Display prediction
st.subheader('Credit Score')
st.write(prediction_proba[1])
