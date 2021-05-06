import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.write("""
	# Pret A Depenser - Credit Dashboard
	""")

# Load processed test data
data = pd.read_csv('test_data.csv')
df = data.drop('TARGET', axis = 1)
df=df.drop('Unnamed: 0', axis=1)
cust_list = df['SK_ID_CURR'].unique()[:10]
main_feat = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_3']


# Input and save Credit Application Number (SK_ID_CURR)
st.sidebar.header('Credit Application Number')
cust_id = st.sidebar.selectbox('Select SK_ID_CURR',cust_list)


# Apply saved model to make predictions
load_clf = joblib.load("trained_model.joblib")
prediction_proba = load_clf.predict_proba(df.loc[df['SK_ID_CURR'] == cust_id])
score = prediction_proba[0][1]*100
thd = joblib.load("trained_model_threshold.joblib")*100


# Display prediction
st.header('Selected customer details')
st.subheader('Credit Score (/100)')
st.write(int(score))

# Display decision
st.subheader('Credit Decision')
if score > thd :
	st.write('Approved')
else :
	st.write('Refused')


# Display Customer general information
st.subheader('Selected customer key Information')
st.dataframe(df[main_feat][df['SK_ID_CURR']==cust_id])


# Display Average information
st.header('Global Customers Key Information')
st.write('0 = min, 1 = avg, 2 = max')

for mf in main_feat:
    st.subheader(mf)
    st.bar_chart([df[mf].min(), df[mf].mean(), df[mf].max()])
