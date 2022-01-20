import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, log_loss
#For the neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import os
import random


"""STEP 1 = LOADING, EDDITING, VISUALISING DATASET"""

#LOADING dataset
filename = 'BTC-USD.csv'
data = pd.read_csv(filename)

#EDITING DATASET
#check for null values
print(data.isnull().sum())
print("Dataframe's shape: ",data.shape)
# get number of unique values for each column
print(data.nunique())

#Visualising Dataset
#as time series
data.plot.line(y="Close", x="Date")
plt.show()

#as multivariable function
d=data.to_numpy()
X = data[ ["Close"] ].to_numpy()#as_matrix()
y = data[ ["Open", "High", "Low", "Volume"] ].to_numpy()
plt.plot( X, y, 'rx', markersize=2 )
plt.show()

data.describe()

data['Open'].plot.kde()

data['High'].plot.kde()

data['Low'].plot.kde()

data['Close'].plot.kde()

data['Adj Close'].plot.kde()

data['Volume'].plot.kde()

"""STEP 2 = CLOSING VALUES: PREPROCESSING-TRAINING-DIAGNOSTICS"""

#normalisation
count=0
for i in range(data.shape[0]):
  if(data['Adj Close'][i]==data['Close'][i]):
    count+=1
print(count)

scaler = MinMaxScaler()
#we dont need column "Date"
#we will also drop column "Adj Close, as it is the same as column "Close"
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
X=data_close[predictors]
data_close.head()

"""# Logistic Regression Model"""

y=data_close['Target']
x_train,x_test,y_train,y_test=train_test_split(X,y,shuffle=False,test_size=0.2)

clf1 = LogisticRegression(random_state=0)
scores=np.array(cross_val_score(clf1, x_train, y_train, cv=5))
c_v_score=np.mean(scores)
print(c_v_score)

clf2 = LogisticRegression(random_state=0,C=0.01,class_weight='balanced')
scores=np.array(cross_val_score(clf2, x_train, y_train, cv=5))
c_v_score=np.mean(scores)
print(c_v_score)

clf3 = LogisticRegression(random_state=0,C=0.01,penalty='elasticnet',solver='saga',l1_ratio=0.0025)
scores=np.array(cross_val_score(clf3, x_train, y_train, cv=5))
c_v_score=np.mean(scores)
print(c_v_score)

pred=clf3.fit(x_train, y_train).predict(x_test)

print("LOSS:",log_loss(y_test, pred))

"""# Linear Regression Model"""

y=np.array(data_close['Actual_Close'])
x_train,x_test,y_train,y_test=train_test_split(X,y,shuffle=False,test_size=0.3)

clf1 = LinearRegression()
scores=np.array(cross_val_score(clf1, x_train, y_train, cv=5))
c_v_score=np.mean(scores)
print(c_v_score)

clf2 = LinearRegression(fit_intercept=False)
scores=np.array(cross_val_score(clf2, x_train, y_train, cv=5))
c_v_score=np.mean(scores)
print(c_v_score)

clf3 = LinearRegression(positive=True)
scores=np.array(cross_val_score(clf3, x_train, y_train, cv=5))
c_v_score=np.mean(scores)
print(c_v_score)

pred=clf2.fit(x_train, y_train).predict(x_test)
print("TEST RMSE:",mean_squared_error(y_test, pred,squared=False))

"""
# Neural Network Model"""

close_values=data_close['Actual_Close']
train,test=train_test_split(close_values,test_size=0.2)
train=np.array(train).reshape(train.shape[0],1)
test=np.array(test).reshape(test.shape[0],1)
train.shape,test.shape

xTrain=[]
yTrain=[]
for i in range(50,train.shape[0]):
  xTrain.append(train[i-50:i])
  yTrain.append(train[i])
xTrain,yTrain=np.array(xTrain),np.array(yTrain)
xTrain.shape,yTrain.shape

xTest=[]
yTest=[]
test=np.array(test)
for i in range(50,test.shape[0]):
  xTest.append(test[i-50:i])
  yTest.append(test[i])
xTest,yTest=np.array(xTest),np.array(yTest)
xTest.shape,xTest.shape

device='cpu'
yTest=torch.from_numpy(yTest).to(device)

xTrain=torch.from_numpy(xTrain).to(device)
yTrain=torch.from_numpy(yTrain).to(device)

xTest=torch.from_numpy(xTest).to(device)
train_set = TensorDataset(xTrain,yTrain) # create your datset
train_loader = DataLoader(train_set,batch_size=32,num_workers=0) #

test_set = TensorDataset(xTest,yTest) # create your datset
val_loader = DataLoader(test_set,batch_size=32,num_workers=0)

def seed_all():
  seed=185
  torch.manual_seed(seed)


  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
seed_all()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            #nn.Linear(50, 50),
            #nn.ReLU(),
            nn.Linear(50,1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        #print("X.shape=",x.shape)
        x=x.view(x.shape[0],1,50)
        logits = self.linear_relu_stack(x)
        return logits

seed_all()
model = NeuralNetwork().to(device)
test=model(torch.rand(32,50,1).to(device))
test.shape

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,weight_decay=3e-4)
epochs=6

for epoch in range(epochs):
  with torch.no_grad():
    for batch,(x,y) in enumerate(train_loader):
      # Compute prediction and loss
      pred = model(x.float())
      loss = loss_fn(pred.view(pred.shape[0],1), y.float())
      loss = torch.sqrt(loss+0.0000001) 
  
pred=model(xTest.float()).detach().cpu().numpy()
pred=pred.reshape(pred.shape[0],1)
y=yTest.detach().cpu().numpy()
print("TEST RMSE=",mean_squared_error(y, pred, squared=False))