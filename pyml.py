from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib as plt

##################### STEP 1 = LOADING, EDDITING, VISUALISING DATASET #####################
#LOADING DATASET
filename = 'dataset/BTC-USD.csv'
data = pd.read_csv(filename)
print( "DataFrame has shape of ", data.shape )
data.head(5)

#EDITING DATASET
#TODO:CHECK IF TAKING INTO CONSIDERATION THE OTHER COLUMNS IS NEEDED
data["Close"].plot().get_figure()
data_editing=data["Close"]
removed_outliers = data_editing.between(data_editing.quantile(.05), data_editing.quantile(.95))
print(str(data_editing[removed_outliers].shape[0]) + "/" + str(data.shape[0]) + " data points remain.")
index_names = data[~removed_outliers].index # INVERT removed_outliers!!
data.drop(index_names, inplace=True)
data.head(5)

#VISUALISING DATASET
#as time series
data.plot.line(y="Close", x="Date")
plt.pyplot.show()
#as a linear function
d=data.to_numpy()
X = data[ ["Close"] ].to_numpy()#as_matrix()
y = data[ ["Open", "High", "Low", "Volume"] ].to_numpy()
print( X.shape, y.shape )
plt.pyplot.plot( X, y, 'rx', markersize=2 )
plt.pyplot.show()

##################### STEP 2 = CLOSING VALUES: PREPROCESSING-TRAINING-DIAGNOSTICS #####################
#PREPROCESSING
#normalisation(without column "Date")
scaler = preprocessing.MinMaxScaler()
names =["Open", "High", "Low", "Volume", "Close"]
data = scaler.fit_transform(data[names])
data = pd.DataFrame(data, columns=names)
data.head()
#keep close data in a separate DataFrame
data_close= data[["Close"]] #needed in order to preserve the actual close value per day
data_close= data_close.rename(columns = {'Close':'Actual_Close'})
#rolling over every two rows, check if the later close value is greater than the preceding one and return 1, otherwise 0
data_close["Target"] = data.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
data_close.head()
#ensure that the prediction of future prices depends on past day's data
data_prev = data.copy()
data_prev = data_prev.shift(1)
data_prev.head()

#TRAINING
# Create training data
predictors = ["Volume", "Open", "High", "Low"]
data_close = data_close.join(data_prev[predictors]).iloc[1:]
data_close.head()
# Train our model
lin_reg = LinearRegression()
x_train, x_test,y_train,  y_test= train_test_split(data_close[predictors], data_close["Actual_Close"])
lin_reg.fit(x_train, y_train)

#DIAGNOSTICS
#TODO:CHECK WANTED PRINTED VALUES
r_sq = lin_reg.score(x_train, y_train)
print('Linear Regression: coefficient of determination:', r_sq)
print('Linear Regression: intercept:', lin_reg.intercept_) #tells how much our model predicts the response when 𝑥_train is zero
print('Linear Regression: slope:', lin_reg.coef_)

