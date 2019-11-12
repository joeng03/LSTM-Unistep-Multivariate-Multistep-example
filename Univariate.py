import pandas as pd
import numpy as np
from io import StringIO
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam

scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv('GS.csv')
df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']
plt.figure(figsize=(16,8))
plt.plot(df['Close'])

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
dataset = new_data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
train= scaled_data[:int(df.shape[0]*0.8)]
valid = scaled_data[int(df.shape[0]*0.8):]

x_train,y_train,x_test,y_test = [],[],[],[]
for i in range(60,train.shape[0]):
    x_train.append(train[i-60:i,0])
    y_train.append(train[i,0])

for z in range(60,valid.shape[0]):
    x_test.append(valid[z-60:z,0])
    y_test.append(valid[z,0])

x_train, y_train,x_test,y_test = np.array(x_train), np.array(y_train),np.array(x_test),np.array(y_test)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
model = Sequential()
model.add(LSTM(units=100,input_shape=(x_train.shape[1],1),return_sequences=True))
model.add(LSTM(units=100))
model.add(Dropout(0.4))
model.add(Dense(1))
ADAM=keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=ADAM)
history = model.fit(x_train,y_train,epochs=50,batch_size=72,validation_data=(x_test,y_test),verbose=1,shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

train= new_data[:int(df.shape[0]*0.8)]
valid = new_data[int(df.shape[0]*0.8):]
valid['Predictions'] = closing_price
plt.xlabel('Date')
plt.ylabel('Share Price')
plt.plot(train['Close'])
plt.plot(valid['Close'])
line2=plt.plot(valid['Predictions'])
import matplotlib.patches as mpatches
patch=mpatches.Patch(color='green', label='Predicted share price')
plt.legend(line2,handles=[patch],loc=2,fontsize=10)
plt.show()
