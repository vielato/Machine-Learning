from sklearn.ensemble import LinerRegression, LogisticRegression
import numpy as np
import pandas as pd
import matplotlib

#load dataset
filename = 'dataset/BTC-USD.csv'
data = pd.read_csv(filename)

#visualisation of the initial dataset
data.plot.line(y="Close", x="Date")
matplotlib.pyplot.show()

#Set up targeted values => Preprocessing: normalization
data_close= data[["Close"]] #needed in order to preserve the actual close value per day
data_close= data_close.rename(columns = {'Close':'Actual_Close'})

#rolling over every two rows, check if the later close value is greater than the preceding one and return 1, otherwise 0
data_close["Target"] = data.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
data_close.head()

#ensure that the prediction of future prices depends on past data
data_prev = data.copy()
data_prev = data_prev.shift(1)
data_prev.head()

# Create training data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data_close = data_close.join(data_prev[predictors]).iloc[1:]
data.head()
