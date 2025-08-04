import pandas as pd 
import numpy as np 
import yfinance as yf 
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st
import matplotlib.pyplot as plt

model=load_model("C:\\Users\\Ravikumar L\\OneDrive\\Desktop\\jupyter projects\\stock price prediction\\Stock Predictions Model.keras")

st.header("Stock Market Price Predictor")

stock=st.text_input("Enter Stock Symbol", 'GOOG')

start = '2014-01-01'
end = '2025-03-31'
data = yf.download(stock, start=start, end=end)

if data.empty:
    st.error("No data found for the given stock symbol. Please try a different one.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

data_train=pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80) : len(data)])

if data_test.empty or data_train.empty:
    st.error("Not enough historical data to proceed.")
    st.stop()

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

past_100days_data = data_train.tail(100)
data_test = pd.concat([past_100days_data,data_test],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('price VS moving average50')
ma_50_days=data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r', label='mov_50_days')
plt.plot(data.Close,'g', label='actual values')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('price Vs moving avg50 Vs moving avg100')
ma_100_days=data.Close.rolling(100).mean()
fig2=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r', label='mov_50_days')
plt.plot(ma_100_days,'b', label='mov_100_days')
plt.plot(data.Close,'g', label='actual values')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('price Vs moving avg100 Vs moving avg200')
ma_200_days=data.Close.rolling(200).mean()
fig3=plt.figure(figsize=(8,6))
plt.plot(ma_200_days,'r', label='mov_200_days')
plt.plot(ma_100_days,'b', label='mov_100_days')
plt.plot(data.Close,'g', label='actual values')
plt.legend()
plt.show()
st.pyplot(fig3)

x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y = np.array(x),np.array(y)

y_predict=model.predict(x)

scaled=1/scaler.scale_

y_predict=y_predict*scaled
y=y*scaled

st.subheader('predicted price Vs actual price')
fig4=plt.figure(figsize=(8,6))
plt.plot(y_predict,'r',label='predicted values')
plt.plot(y,'g', label='actual values')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig4)