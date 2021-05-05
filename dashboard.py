import streamlit as st
import pandas as pd
import joblib


st.write("""
# Credit Score App

This app provides information regarding credit decision based on customer scoring

""")

# Load processed test data
data = pd.read_csv('test_data.csv')
df = data.drop('TARGET', axis = 1)
df=df.drop('Unnamed: 0', axis=1)
cust_list = df['SK_ID_CURR'].unique()[:10]


# Input and save Credit Application Number (SK_ID_CURR)
st.sidebar.header('Credit Application Number')
cust_id = st.sidebar.selectbox('Select SK_ID_CURR',cust_list)

# Apply saved model to make predictions
load_clf = joblib.load("trained_model.joblib")
prediction_proba = load_clf.predict_proba(df.loc[df['SK_ID_CURR'] == cust_id])
score = prediction_proba[0][1]*100
thd = joblib.load("trained_model_threshold.joblib")*100


# Display prediction
st.subheader('Credit Score (/100)')
st.write(int(score))

# Display decision
st.subheader('Credit Decision')

if score > thd :
	st.write('Approved')

else :
	st.write('Refused')


# Display Customer general information
st.subheader('Customer Key Information')
main_feat = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_3', 'DAYS_BIRTH','AMT_ANNUITY']
st.dataframe(df[main_feat][df['SK_ID_CURR']==cust_id])

# Display Average information
st.subheader('Overall Key Information')
st.write("""
Average
""")
st.dataframe(df[main_feat].mean())

st.write("""

Standard Deviation
""")
st.dataframe(df[main_feat].std())