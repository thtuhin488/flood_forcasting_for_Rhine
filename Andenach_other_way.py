import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# load data and arrange into Pandas dataframe
cols=["Evaporation","Precipitation","Temperature","Discharge"]
df=pd.read_excel("sheet 1.xlsx",usecols=cols)



#Split into parameters and target (Discharge)
X = df.drop('Discharge', axis = 1)
y = df['Discharge']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 20)


#Scale data, otherwise model will fail.
#Standardize features by removing the mean and scaling to unit variance

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# define the model
#Experiment with deeper and wider networks
model = Sequential()
model.add(Dense(128, input_dim=3, activation='relu'))
model.add(Dense(64, activation='relu'))
#Output layer
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()


history = model.fit(X_train_scaled, y_train, validation_split=0.33, epochs =100)



predictions = model.predict(X_test_scaled[:5])
print("Predicted values are: ", predictions)
print("Real values are: ", y_test[:5])


mse_neural, mae_neural = model.evaluate(X_train_scaled,y_train)
print('Mean squared error from neural net for train: ', mse_neural)
print('Mean absolute error from neural net for train: ', mae_neural)

mse_neural, mae_neural = model.evaluate(X_test_scaled,y_test)
print('Mean squared error from neural net for test: ', mse_neural)
print('Mean absolute error from neural net for test: ', mae_neural)
