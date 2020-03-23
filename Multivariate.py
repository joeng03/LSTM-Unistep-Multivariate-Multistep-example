import pandas as pd
import numpy as np
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from technicalindicators import Technical
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
TI=TechnicalIndicators()
close_data=TI.close_data[['4. close']]
macd_data=TI.macd_data
rsi_data=TI.rsi_data
bbands_data=TI.bbands_data
#sma_data=TI.sma_data
dataset = pd.concat([macd_data,rsi_data,bbands_data,close_data], axis=1,sort=True).reindex(macd_data.index)
dataset=dataset.drop(dataset.index[len(dataset)-1])
#dataset = pd.concat([macd_data,close_data,], axis=1, sort=True).reindex(macd_data.index)
close_data = dataset[['4. close']]
X=dataset.drop(dataset.index[len(dataset)-1])
y=close_data.drop(close_data.index[0])
#y=np.ravel(y)
values_x=X.values
values_y=y.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_X = scaler.fit_transform(values_x)
scaled_data_y = scaler.fit_transform(values_y)
X_train=scaled_data_X[:int(X.shape[0]*0.8)]
X_test= scaled_data_X[int(y.shape[0]*0.8):]
y_train=scaled_data_y[:int(X.shape[0]*0.8)]
y_test=scaled_data_y[int(y.shape[0]*0.8):]

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1])
y_train=y_train.reshape(y_train.shape[0],1,1)
y_test=y_test.reshape(y_test.shape[0],1,1)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
a_train,b_train,c_test,d_test =[],[],[],[]
for i in range(60,y_train.shape[0]):
    a_train.append(X_train[i-60:i,0])
    b_train.append(y_train[i,0])
for x in range(60,y_test.shape[0]):
    c_test.append(X_test[x-60:x,0])
    d_test.append(y_test[x,0])
a_train, b_train,c_test,d_test = np.array(a_train), np.array(b_train),np.array(c_test),np.array(d_test)
a_train = np.reshape(a_train, (a_train.shape[0],a_train.shape[1],8))
b_train=np.reshape(b_train,(b_train.shape[0],b_train.shape[1]))
c_test=np.reshape(c_test,(c_test.shape[0],c_test.shape[1],8))
d_test=np.reshape(d_test,(d_test.shape[0],d_test.shape[1]))

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import keras
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(a_train.shape[1],a_train.shape[2])))
model.add(LSTM(units=100))
model.add(Dropout(0.4))
model.add(Dense(1))
ADAM=keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=ADAM)
history = model.fit(a_train,b_train,epochs=50,batch_size=60,validation_data=(c_test,d_test),verbose=1,shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

from numpy import concatenate
yhat = model.predict(c_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
inv_yhat = concatenate((yhat,X_test[60:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test[60:],X_test[60:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_yhat)
plt.plot(yhat)
plt.show()





