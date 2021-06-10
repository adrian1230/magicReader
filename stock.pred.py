import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk
import datetime as dt
import csv
import os
from model import *
from keras.models import *
from sklearn.preprocessing import MinMaxScaler

tod = dt.date.today()

pm1 = int(tod.strftime("%m"))

prev = ""

if pm1 > 3:
    pm1 = pm1 - 3
    pm1 = str(pm1)
    prev = tod.strftime("%Y-{}-%d").format(pm1)

if pm1 == 1:
    pm1 = str(10)
    y = int(tod.strftime("%Y"))-1
    y = str(y)
    prev = tod.strftime("{}-{}-%d").format(y,pm1)

if pm1 == 2:
    pm1 = str(11)
    y = int(tod.strftime("%Y"))-1
    y = str(y)
    prev = tod.strftime("{}-{}-%d").format(y,pm1)

if pm1 == 3:
    pm1 = str(12)
    y = int(tod.strftime("%Y"))-1
    y = str(y)
    prev = tod.strftime("{}-{}-%d").format(y,pm1)

today = tod.strftime("%Y-%m-%d")

st.write("""
# Visualize and Predict One Stock from Yahoo Finance
""")

st.header("Enter the stock code here!")

st.subheader("Example Input: AAPL; 2010-03-14, 2015-03-14")

st.text("Train from 2010-03-14 to 2015-03-14")

stock_codes = "GE; {}, {}".format(prev,today)

lit = st.text_input("Stock & History Input", stock_codes)

st.text("Fair Tip -")
st.text("Whenever you want to predict the stock price, you can:")
st.text("Use the first provided model, which trained on a year of data, or")
st.text("retrain it using the most recent 3 months of data")

hist  = lit.split(';')[1].split(',')

lit = lit.split(';')[0].split(',')

for k in range(len(hist)):
    hist[k] = hist[k].strip()

for i in range(len(lit)):
    lit[i] = lit[i].strip()

if len(lit) > 1:
    raise ValueError("More than 1")

st.write("""
***
""")

st.header("Your selected stock: {}".format(lit[0]))

st.subheader("Training or Application")

choice = st.text_input("Training or Application; 0 vs 1: ", 1)

choice = int(choice)

# if choice < 0 or choice > 1:
#     raise ValueError("other than 1 or 0")

def option_(choice):
    if choice == 0:
        return "Training"
    elif choice == 1:
        return "Application"
    else:
        raise ValueError("No other option beyond prediction and training")

st.subheader("You chose {}".format(option_(choice)))

data = yf.Ticker(lit[0])
df = data.history(start=hist[0],end=hist[1])
open_tick = []
open_tick.append(df.Open)
open_tick = np.array(open_tick).reshape(-1)
date_tick = []
date_tick.append(df.index)
date_tick = np.array(date_tick).reshape(-1)
for r in range(len(date_tick)):
    date_tick[r] = str(pd.to_datetime(str(date_tick[r]))).split(' ')[0]
close_tick = []
close_tick.append(df.Close)
close_tick = np.array(close_tick).reshape(-1)
high_tick = []
high_tick.append(df.High)
high_tick = np.array(high_tick).reshape(-1)
low_tick = []
low_tick.append(df.Low)
low_tick = np.array(low_tick).reshape(-1)
volume_tick = []
volume_tick.append(df.Volume)
volume_tick = np.array(volume_tick).reshape(-1)
dividends_tick = []
dividends_tick.append(df.Dividends)
dividends_tick = np.array(dividends_tick).reshape(-1)

Closingchart = pd.DataFrame(close_tick)

DateChart = pd.DataFrame(date_tick)

Openchart = pd.DataFrame(open_tick)

Highchart = pd.DataFrame(high_tick)

Lowchart = pd.DataFrame(low_tick)

Volumechart = pd.DataFrame(volume_tick,)

Dividendchart = pd.DataFrame(dividends_tick)

Chart = pd.concat([DateChart,Openchart,Highchart,Lowchart,Volumechart,Closingchart, Dividendchart],axis=1)

Chart.columns = ['Date','Open','High','Low','Volume','Dividends','Close']

Chart.to_csv('Now.csv')

train_normal = nor(Chart) # from model.py
x_train, y_train = build(train_normal,25,25) # from model.py
x_train, y_train = shuf(x_train,y_train) # from model.py
x_train, y_train, x_val, y_val = spd(x_train,y_train,0.1) # from model.py
y_train = y_train[:,:,np.newaxis]
y_val = y_val[:,:,np.newaxis]

if choice == 0:
    process = 0 
    while process != 1:
        st.header("Begin training now")
        st.text("Training ...")
        model = mmstack(x_train.shape)
        callback = EarlyStopping(monitor="loss",patience=10,verbose=1,mode="auto")
        summary = model.summary()
        with open('summary.txt','w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        past = model.fit(x_train,y_train,epochs=1000,batch_size=128,validation_data=(x_val,y_val))
        st.write("""
        ***
        """)
        arr = []
        st.subheader("Model Summary:")
        with open('summary.txt','r') as fh:
            lines = fh.readlines()
            for m in lines:
                arr.append(m)
        fh.close()
        for u in arr:
            st.text("{}".format(u))
        model.save("sp.h5")
        process += 1
    st.header("Training finished")
    st.write("""
        ***
    """)
if choice == 1:
    mms = MinMaxScaler()
    loaded = load_model('./sp.h5')
    latest5 = Chart.drop(["Date","Close"],axis=1).tail(25)
    latest5 = mms.fit_transform(latest5)
    latest5 = np.array(latest5)
    latest5 = latest5.reshape((1,latest5.shape[0],latest5.shape[1]))
    pred = loaded.predict(latest5)
    pred = pred.reshape(-1,5)
    last = mms.inverse_transform(pred)
    st.subheader("the prediction of {} for the following 5 days".format(lit[0]))
    # st.text("Based on only the stock market open day")
    # st.text("The model was based on the {} stock data from {} to {}".format(lit[0],prev,today))
    with open('five.csv','w') as csv_:
        writer = csv.writer(csv_,delimiter=',')
        name = ["Open","High","Low","Volume","Dividends"]
        writer.writerow([j for j in name])
        for f in range(len(last)):
            writer.writerow(last[f])
    csv_.close()
    df = pd.read_csv('./five.csv')
    df.index = ["Day {}".format(r+1) for r in range(5)]
    st.dataframe(df)
    
