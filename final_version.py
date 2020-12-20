# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:31:11 2020

@author: sihan
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 07:43:49 2020

@author: sihan
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 07:21:57 2020

@author: sihan
"""
#%%
import os
from tkinter import *
import tkinter
from tkinter.filedialog import askopenfilename
import pandas as pd
import sys
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers.recurrent_v2 import _CPU_DEVICE_NAME

#%%

history=0
canvas=0
canvas2=0


#%%
def _quit():
    """
    destroy the root
    """
    root.quit()
    root.destroy()

def import_csv_data():
    """
    Read the .csv file in tinket
    """
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    global df
    df = pd.read_csv(csv_file_path,  sep=';',
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')

def handle_missing_value():
    """
    Fill the missing value with the mean value of the corresponding column
    """
    droping_list_all=[]
    N = df.shape[1]
    for j in range(0,N):
        if not df.iloc[:, j].notnull().all():
            droping_list_all.append(j)
        #print(df.iloc[:,j].unique())
    N2 = len(droping_list_all)
    for j in range(0,N2):
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

def plot():
    fig = Figure(figsize = (5,5), dpi=200)

    y = [i ** 2 for i in range(101)]

    plot1 = fig.add_subplot(111)
    plot1.plot(y)

    # global canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    canvas.get_tk_widget().pack()

def _clear():
    """
    Try to clear the plot.
    """
    # for item in canvas.get_tk_widget().find_all():
    #     canvas.get_tk_widget().delete(item)
    canvas.delete('all')

def plot1(column_number=1, resample_rate = 'D'):
    """
    Plot how the feature change with time
    """
    
    
    # global canvas
    # fig = Figure(figsize=(2,2), dpi=200)
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(14, 7))
    df.iloc[:,column_number].resample(resample_rate).sum().plot(title='Global_active_power resampled over day for sum',ax=ax)
    plt.show()
    # ax.show()

    
    # canvas = FigureCanvasTkAgg(fig, master=root)
    # canvas.get_tk_widget().pack()

    # toolbar = NavigationToolbar2Tk(canvas, root)
    # toolbar.update()

    # canvas.get_tk_widget().pack()
        
def plot2(n=0):
    
    handle1 = plt.plot(history.history['loss'])[0]
    handle2 = plt.plot(history.history['val_loss'])[1]
    
    plt.xlabel('epoch')
    plt.ylbale('val_loss')
    plt.legend(handles = [handle1, handle2], Label=['loss', 'val_loss'])
    
    # # global canvas2
    # fig = Figure(figsize=(2,2), dpi=200)
    # ax = fig.add_subplot(111)
    # if n==0:
    #     ax.plot(history.history['loss'])
    #     ax.set_xlabel('epoch')
    #     ax.set_ylabel('loss')
    #     ax.set_title('model loss')
    # else:
    #     ax.plot(history.history['val_loss'])
    #     ax.set_xlabel('val_loss')
    #     ax.set_ylabel('loss')
    #     ax.set_title('model loss')
    
    # global canvas
    # canvas = FigureCanvasTkAgg(fig, master=root)
    # # canvas2.get_tk_widget().pack()

    # # toolbar = NavigationToolbar2Tk(canvas2, root)
    # # toolbar.update()

    # # canvas2.get_tk_widget().pack()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def train_model(train_X, train_y, test_X, test_y):
    global model
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',  optimizer = 'adam')
    
    checkpoint_path = 'cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only = True, verbose=1)
    
    global history
    history = model.fit(train_X, train_y, epochs = 20, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[cp_callback])
    
    
    # global weight
    # model.get_weights()
    
    
def dataset_processing():
    
    global df_resample
    # resample the data
    df_resample = df.resample('h').mean()
    values = df_resample.values
    
    global scaler
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
    
    
    
    # split into train and test sets
    values = reframed.values
    
    n_train_time = 365*24
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    global train_X, test_X, train_y, test_y
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

def plot_train(n=0):
    fig = Figure(figsize=(2,2), dpi=200)
    plot1 = fig.add_subplot(111)
    if n==0:
        plot1.plot(history.history['loss'])
        plot1.set_xlabel('epoch')
        plot1.set_ylabel('loss')
        plot1.set_title('model loss')
    else:
        plot1.plot(history.history['val_loss'])
        plot1.set_xlabel('val_loss')
        plot1.set_ylabel('loss')
        plot1.set_title('model loss')

    canvas = FigureCanvasTkAgg(plot1, master=root)
    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    canvas.get_tk_widget().gpack()

def prediction():
    """
    This function is used to use the model to do prediction
    """
    global test_X
    y_predict = model.predict(test_X)
    N = df.shape[1]
    
    test_X = test_X.reshape((test_X.shape[0], N))
    
    inv_y_predict = np.concatenate((y_predict, test_X[:, -(N-1):]), axis=1)
    inv_y_predict = scaler.inverse_transform(inv_y_predict)
    
    inv_y_predict = inv_y_predict[:,0]
    
    global test_y 
    test_y = test_y.reshape((len(test_y), 1))
    
    inv_y = np.concatenate((test_y, test_X[:, -(N-1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    
    rmse = np.sqrt(mean_squared_error(inv_y, inv_y_predict))
    print('Test RMSE: %.3f' % rmse)

    time = [x for x in range(200)]
    plt.plot(time, inv_y[:200], marker = '.', label = 'real value')
    plt.plot(time, inv_y_predict[:200], 'r', label='prediction')
    plt.ylabel('Global_active_power', size=15)
    plt.xlabel('time step')
    plt.legend()
    plt.show()

    

#%% Input data wget
root = Tk()
root.title("Sequential data analysis")

Label(root, text='File Path').pack()
v = StringVar()
entry = Entry(root, textvariable=v).pack()
Button(root, text='Browse Data Set',command=import_csv_data).pack()
Button(root, text='Close',command=root.destroy).pack()

#%%
Button(root, text="Missing value handle", command=lambda: handle_missing_value()).pack()
e1 = Entry(root, borderwidth=15)
e1.pack()
e1.insert(0, "Insert the desired column")
e2 = Entry(root, borderwidth=15)
e2.pack()
e2.insert(0, "Insert the desired resample rate")


plot_button = Button(root, height=2, width=10, text='Plot', command=lambda:plot1(int(e1.get()), str(e2.get())))
plot_button.pack()
# photo_clear = Button(root, height=2, width=10, text='Clear', command=_clear)
# photo_clear.pack()



Button(root, text='quit', command=_quit).pack()

Button(root, text='data_preprocessing',command=dataset_processing).pack()


Button(root, text='train', command=lambda: train_model(train_X, train_y, test_X, test_y)).pack()
Button(root, text="Plot the train result", command=lambda: plot2()).pack()

Button(root, text="Predict test_X", command=lambda: prediction()).pack()



#%%
root.mainloop()
