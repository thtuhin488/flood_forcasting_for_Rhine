import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import sklearn as sklearn

#importing data

excel_file = 'C:/Users/thtuh/Desktop/project/Final Data/CDC/for python/sheet 1.xlsx'
Dt = pd.read_excel(excel_file,usecols=['Date'])
D = pd.read_excel(excel_file,usecols=['Discharge'])

#plt.plot(D)



#Convert pandas dataframe to numpy array

dataset = D.values
dataset = dataset.astype('float32') #COnvert values to float

# Normalization is optional but recommended for neural network as certain 
# activation functions are sensitive to magnitude of numbers. 
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# X and Y. We need to transform our data into something that looks like X and Y values.
# This way it can be trained on a sequence rather than indvidual datapoints.
# Let us convert into n number of columns for X where we feed sequence of numbers
# then the final column as Y where we provide the next number in the sequence as output.
# So let us convert an array of values into a dataset matrix

# seq_size is the number of previous time steps to use as
# input variables to predict the next time period.

# creates a dataset where X is the number of discharge at a given time (t, t-1, t-2...)
# and Y is the number of discharge at the next time (t + 1).

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size - 1):
        # print(i)
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)

'''
trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))



'''

seq_size = 5 # Number of time steps to look back

# Larger sequences (look further back) may improve forecasting.
trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)
#Compare trainX and dataset. You can see that X= values at t, t+1 and t+2


#whereas Y is the value that follows, t+3 (since our sequence size is 3)

print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))

#Input dimensions are... (N x seq_size)
print('Build deep model...')
# create and fit dense model
model = Sequential()
model.add(Dense(64, input_dim=seq_size, activation='relu'))
model.add(Dense(32 , activation='relu'))
model.add(Dense(32 , activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
print(model.summary())

#model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

model.fit(trainX, trainY, validation_data=(testX, testY),
          verbose=2, epochs=100)
# make predictions
trainPredict1 = model.predict(trainX)
testPredict1 = model.predict(testX)

predictions = model.predict(testX[:5])
print("Predicted values are: ", predictions)
print("Real values are: ", test[:5])

#Scatter plot
x=testPredict1
y=test[:740]

lineStart = x.min()
lineEnd = y.max()

plt.figure()
plt.scatter(x, y, color = 'k', alpha=0.5)
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
plt.show()
plt.xlabel('predicted')
plt.ylabel('observed')
plt.legend(loc="upper right")
plt.show()


# Estimate model performance

#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.

trainPredict = scaler.inverse_transform(trainPredict1)
trainY_inverse = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict1)
testY_inverse = scaler.inverse_transform([testY])


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY_inverse[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


#Neural network - from the current code
#mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
mse_neural, mae_neural = model.evaluate(trainX,trainY)
print('Mean squared error from neural net for train: ', mse_neural)
print('Mean absolute error from neural net for train: ', mae_neural)

mse_neural, mae_neural = model.evaluate(testX,testY)
print('Mean squared error from neural net for test: ', mse_neural)
print('Mean absolute error from neural net for test: ', mae_neural)

print('R^2 Value from train: ', sklearn.metrics.r2_score(trainY_inverse[0], trainPredict[:,0]))
print('R^2 Value from test: ', sklearn.metrics.r2_score(testY_inverse[0], testPredict[:,0]))


###########
# shift train predictions for plotting
# shift the predictions so that they align on the x-axis with the original dataset.
import math, numpy
math.isnan(numpy.nan)

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = math.isnan(numpy.nan)

trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict


# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)

testPredictPlot[:,:] = math.isnan(numpy.nan)
testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1,:] = testPredict

# plot baseline and predictions

plt.plot(Dt[1827:2191],scaler.inverse_transform(dataset[1827:2191]),color='black',linewidth =.8,label='Observed Discharge')
#plt.plot(Dt,trainPredictPlot,linewidth =.6,color='orange')
plt.plot(Dt[1827:2191], testPredictPlot[1827:2191],linewidth =.6,color='red',label='Predicted Discharge')
plt.show()
plt.xlabel('Date')
plt.ylim(600,6400)
plt.ylabel('Discharge(m3/s')
plt.legend(loc="upper right")
plt.show()


'''''

plt.plot(Dt[1460:1827],scaler.inverse_transform(dataset)[1460:1827], color='black',linewidth =.8,label='Observed Discharge')
plt.plot(Dt[1460:1827], testPredictPlot[1460:1827],color='r',linewidth = 1,label='Predicted Discharge')

#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.xlim(2016,2017)
plt.ylim(600,6400)
plt.xlabel('Date')
plt.ylabel('Discharge(m3/s')
plt.title()
plt.legend(loc="upper right")
plt.show()
#plt.xlim(2016,2018)
'''''
################### Other Ml Models
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

### Linear regression
lr_model = linear_model.LinearRegression()
lr_model.fit(trainX,trainY)
y_pred_lr = lr_model.predict(testX)
mse_lr = mean_squared_error(testY, y_pred_lr)
mae_lr = mean_absolute_error(testY, y_pred_lr)
RMSE_lr = math.sqrt(mean_squared_error(testY, y_pred_lr))

print('test Score: %.2f RMSE' % (RMSE_lr))

print('Mean squared error from linear regression for test: ', mse_lr)
print('Mean absolute error from linear regression for test: ', mae_lr)

print('R^2 Value from test: ', sklearn.metrics.r2_score(testY, y_pred_lr))

lr_model.fit(trainX,trainY)
yt_pred_lr = lr_model.predict(trainX)
mse_lrt = mean_squared_error(trainY, yt_pred_lr)
mae_lrt = mean_absolute_error(trainY, yt_pred_lr)
RMSE_lrt = math.sqrt(mean_squared_error(trainY, yt_pred_lr))

print('train Score: %.2f RMSE' % (RMSE_lrt))
print('Mean squared error from linear regression for train: ', mse_lrt)
print('Mean absolute error from linear regression for train: ', mae_lrt)
print('R^2 Value from train: ', sklearn.metrics.r2_score(trainY, yt_pred_lr))

### Decision tree
tree = DecisionTreeRegressor()
tree.fit(trainX, trainY)
y_pred_tree = tree.predict(testX)
mse_dt = mean_squared_error(testY, y_pred_tree)
mae_dt = mean_absolute_error(testY, y_pred_tree)
RMSE_dt = math.sqrt(mean_squared_error(testY, y_pred_tree))

print('R^2 Value from test using decision tree: ', sklearn.metrics.r2_score(testY,y_pred_tree ))
print('test Score: %.2f RMSEusing decisin tree' % (RMSE_dt))
print('Mean squared error using decision tree: ', mse_dt)
print('Mean absolute error using decision tree: ', mae_dt)


yt_pred_tree = tree.predict(trainX)
mse_dtt = mean_squared_error(trainY, yt_pred_tree)
mae_dtt = mean_absolute_error(trainY,yt_pred_tree)
RMSE_dtt = math.sqrt(mean_squared_error(trainY, yt_pred_tree))

print('train Score: %.2f RMSE' % (RMSE_dtt))
print('Mean squared error from decision tree for train: ', mse_dtt)
print('Mean absolute error from decision tree for train: ', mae_dtt)
print('R^2 Value from train using decision tree: ', sklearn.metrics.r2_score(trainY,  yt_pred_tree))
##############################################
#Random forest.
#Increase number of tress and see the effect
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 30, random_state=30)
model.fit(trainX, trainY)

y_pred_RF = model.predict(testX)

mse_RF = mean_squared_error(testY, y_pred_RF)
mae_RF = mean_absolute_error(testY, y_pred_RF)
RMSE_RF = math.sqrt(mean_squared_error(testY, y_pred_RF))

print('R^2 Value from test using randomforest ', sklearn.metrics.r2_score(testY,y_pred_RF))
print('test Score: %.2f RMSE' % (RMSE_RF))
print('Mean squared error using Random Forest: ', mse_RF)
print('Mean absolute error Using Random Forest: ', mae_RF)

yt_pred_RF = model.predict(trainX)
mse_RFt = mean_squared_error(trainY, yt_pred_RF)
mae_RFt = mean_absolute_error(trainY,yt_pred_RF)
RMSE_RFt = math.sqrt(mean_squared_error(trainY, yt_pred_RF))

print('train Score: %.2f RMSE' % (RMSE_RFt))
print('Mean squared error from random forest for train: ', mse_RFt)
print('Mean absolute error from random forest for train: ', mae_RFt)
print('R^2 Value from train for random forest: ', sklearn.metrics.r2_score(trainY,  yt_pred_RF))
