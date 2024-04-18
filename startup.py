import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import joblib

data = pd.read_csv('startUp(2).csv')
model = joblib.load('startUpModel.pkl')

st.markdown("<h1 style = 'color: #974D5F; text-align: center; font-family: helvetica'>STARTUP PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #CA677E; text-align: center; font-family: cursive '>Built By IHEMEGBULEM GODSTIME</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com.png')

st.markdown("<h4 style = 'margin: -30px; color: #974D5F; text-align: center; font-family: helvetica '>Project Overview</h4>", unsafe_allow_html = True)

st.write("The goal of this project is to develop a predictive model that assesses the profitability of startup companies. By leveraging machine learning techniques, we aim to provide insights into the factors influencing a startup's financial success, empowering stakeholders to make informed decisions.")

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width= True)

st.sidebar.image('pngwing.com(1).png', caption = 'Welcome Dear User')

rd_spend = st.sidebar.number_input('Research and Development')
admin = st.sidebar.number_input('Administration Expense')
mkt_exp = st.sidebar.number_input('Marketing Expense')

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'margin: -30px; color: #974D5F; text-align: center; font-family: helvetica '>Input Variable</h4>", unsafe_allow_html = True)

inputs = pd.DataFrame()
inputs['R&D Spend'] = [rd_spend]
inputs['Administration'] = [admin]
inputs['Marketing Spend'] = [mkt_exp]

st.dataframe(inputs, use_container_width= True)

# import the transformers 
rd_spend_scale = joblib.load('rd_spend_scale')
mgt_scale = joblib.load('mgt_scale')
mkt_scale = joblib.load('mkt_scale')

# Transforming
inputs['R&D Spend'] = rd_spend_scale.transform(inputs[['R&D Spend']])
inputs['Administration'] = mgt_scale.transform(inputs[['Administration']])
inputs['Marketing Spend'] = mkt_scale.transform(inputs[['Marketing Spend']])

st.markdown("<br>", unsafe_allow_html= True)

prediction_button = st.button('Predict Profitability')
if prediction_button:
   predicted = model.predict(inputs)
   st.success(f'The profit predicted for your company is {predicted[0].round(2)}')