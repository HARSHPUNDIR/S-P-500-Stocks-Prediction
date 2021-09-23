# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

trainingSet = pd.read_csv('all_stocks_5yr.csv')
company = input('Enter Company Name : ')
trainingSet=trainingSet[trainingSet.Name == company]
trainingSet = trainingSet.iloc[:,1:2].values[0:1000]
l = (trainingSet.shape)
sc = MinMaxScaler()
trainingSet = sc.fit_transform(trainingSet)
xTrain = trainingSet[0:l[0]-1]
yTrain = trainingSet[1:l[0]]
xTrain = np.reshape(xTrain, (l[0]-1, 1, 1 ))
model = Sequential()
model.add(LSTM(units = 4, activation = 'sigmoid', input_shape=(None, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(x =xTrain , y =yTrain , batch_size=32, epochs=200)
model.save_weights('LSTM_Model.h5')


#Test Set
testSet = pd.read_csv('all_stocks_5yr.csv')
testSet = testSet[testSet.Name == company][1000:1500]
realStockPrice = testSet.iloc[:, 1:2].values
l = (testSet.shape)
inputs = realStockPrice[0:l[0]]
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (l[0],1,1))
prediction = model.predict(inputs)
prediction = sc.inverse_transform(prediction)
#Visualization the results
plt.plot(realStockPrice, color = 'Red', label='RealPrice')
plt.plot(prediction, color = 'Blue', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price')
plt.legend()
plt.show()
predicted = prediction[-1]
real = realStockPrice[-1]

if predicted > real:
    print('Under Priced')
elif predicted < real:
    print ('Over Priced')
else:
    print ('Correct Proced')
