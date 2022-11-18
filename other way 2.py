import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import f1_score

cols=["Evaporation","Precipitation","Temperature","Discharge"]
df=pd.read_excel("sheet 1.xlsx",usecols=cols)

y = df['Discharge']
X = df.drop('Discharge', axis=1)
scaler = RobustScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x=  tf.keras.layers.Dense(8, activation='relu')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
EPOCHS = 100
BATCH_SIZE = 32
history = model.fit(X_train,y_train,validation_split=0.2,epochs=EPOCHS,batch_size=BATCH_SIZE, verbose=1)



plt.figure(figsize=(14, 10))
plt.plot(range(EPOCHS), history.history['loss'], color='b')
plt.plot(range(EPOCHS), history.history['val_loss'], color='r')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()
