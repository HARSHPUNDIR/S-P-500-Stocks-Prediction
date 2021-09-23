# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
trainingSet = pd.read_csv('all_stocks_5yr.csv')

for i in trainingSet['Name'].unique():
    sc = MinMaxScaler()
    trainingSet=trainingSet[trainingSet.Name == i]
    l = (trainingSet.shape)
    trainingSet = trainingSet[0:int(l[0]/2)]
    testSet = trainingSet[int(l[0]/2):l[0]]
    l=[int(l[0]/2)]
    if l[0] > 100:
        trainingSetOpen = trainingSet.iloc[:,1:2].values
        trainingSetOpen = sc.fit_transform(trainingSetOpen)
        xTrainOpen = trainingSetOpen[0:l[0]-1]
        yTrainOpen = trainingSetOpen[1:l[0]]
        xTrainOpen = np.reshape(xTrainOpen, (l[0]-1, 1, 1 ))

        trainingSetClose = trainingSet.iloc[:,4:5].values
        trainingSetClose = sc.fit_transform(trainingSetClose)
        xTrainClose = trainingSetClose[0:l[0]-1]
        yTrainClose = trainingSetClose[1:l[0]]
        xTrainClose = np.reshape(xTrainClose, (l[0]-1, 1, 1 ))

        trainingSetHigh = trainingSet.iloc[:,2:3].values
        trainingSetHigh = sc.fit_transform(trainingSetHigh)
        xTrainHigh = trainingSetHigh[0:l[0]-1]
        yTrainHigh = trainingSetHigh[1:l[0]]
        xTrainHigh = np.reshape(xTrainHigh, (l[0]-1, 1, 1 ))

        trainingSetLow = trainingSet.iloc[:,3:4].values
        trainingSetLow = sc.fit_transform(trainingSetLow)
        xTrainLow = trainingSetLow[0:l[0]-1]
        yTrainLow = trainingSetLow[1:l[0]]
        xTrainLow = np.reshape(xTrainLow, (l[0]-1, 1, 1 ))

        trainingSetVolume = trainingSet.iloc[:,5:6].values
        trainingSetVolume = sc.fit_transform(trainingSetVolume)
        xTrainVolume = trainingSetVolume[0:l[0]-1]
        yTrainVolume = trainingSetVolume[1:l[0]]
        xTrainVolume = np.reshape(xTrainVolume, (l[0]-1, 1, 1 ))


        model = Sequential()
        model.add(LSTM(units = 4, activation = 'sigmoid', input_shape=(None, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss = 'mean_squared_error')
        model.fit(x =xTrainLow, y =yTrainLow , batch_size=32, epochs=200)
        #model.save_weights('D:\Drive\projects\computer science\python\Share Market\Model\\'+i+'.h5')

        #Test Set
        realStockPrice = testSet.iloc[:, 3:4].values
        inputs = realStockPrice
        inputs = sc.transform(inputs)
        inputs = np.reshape(inputs, (len(realStockPrice),1,1))

        prediction = model.predict(inputs)

        #now the predicted output is scaled, we will apply the reverse transform method to get the actual predicted prices
        prediction = sc.inverse_transform(prediction)

        #Visualization the results
        plt.plot(realStockPrice, color = 'Red', label='RealPrice')
        plt.plot(prediction, color = 'Blue', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Stock Price')
        plt.legend()
        plt.show()
