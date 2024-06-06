# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from DataGeneratorSeq import DataGeneratorSeq
import math
import time


data_source = 'alphavantage' # alphavantage or kaggle

def retriving_data(data_source):
    # ====================== Loading Data from Alpha Vantage =================================
    df = None
    
    if data_source == 'alphavantage':
        # ====================== Loading Data from Alpha Vantage ==================================

        api_key = '29K27P2WQPSDBOHY'

        # American Airlines stock market prices
        ticker = "AAL"

        # JSON file with all the stock market data for AAL from the last 20 years
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

        # Save data to this file
        file_to_save = 'stock_market_data-%s.csv'%ticker

        # If you haven't already saved data,
        # Go ahead and grab the data from the url
        # And store date, low, high, volume, close, open values to a Pandas DataFrame
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1
            print('Data saved to : %s'%file_to_save)        
            df.to_csv(file_to_save)

        # If the data is already there, just load it from the CSV
        else:
            print('File already exists. Loading data from CSV')
            df = pd.read_csv(file_to_save)

    else:

        # ====================== Loading Data from Kaggle ==================================
        # You will be using HP's data. Feel free to experiment with other data.
        # But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization
        df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
        print('Loaded data from the Kaggle repository')
    return df

def data_exploration(df):
    # Sort DataFrame by date
    df = df.sort_values('Date')

    # Double check the result
    df.head()

def data_visualization(df):
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()

# def splitting_data(df):
    # First calculate the mid prices from the highest and lowest
    high_prices = df.loc[:,'High'].to_numpy()
    low_prices = df.loc[:,'Low'].to_numpy()
    mid_prices = (high_prices+low_prices)/2.0
    
    train_data = mid_prices[:int(len(mid_prices)*0.8)]
    test_data = mid_prices[int(len(mid_prices)*0.8):]

    return train_data, test_data

def data_normalization1(train_data, test_data):
    # Scale the data to be between 0 and 1
    # When scaling remember! You normalize both test and train data with respect to training data
    # Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)
    # Train the Scaler with training data and smooth data
    smoothing_window_size = 2500
    for di in range(0, len(train_data), smoothing_window_size):
        if di + smoothing_window_size < len(train_data):
            scaler.fit(train_data[di:di+smoothing_window_size,:])
            train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
        else:
            scaler.fit(train_data[di:,:])
            train_data[di:,:] = scaler.transform(train_data[di:,:])

    # Reshape both train and test data
    train_data = train_data.reshape(-1)

    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)
    # Now perform exponential moving average smoothing
    # So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1
    for ti in range(len(train_data)):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data,test_data],axis=0)

    return train_data, test_data, all_mid_data

def splitting_data(df):
    # First calculate the mid prices from the highest and lowest
    high_prices = df.loc[:,'High'].to_numpy()
    low_prices = df.loc[:,'Low'].to_numpy()
    mid_prices = (high_prices+low_prices)/2.0
    
    train_data = mid_prices[:int(len(mid_prices)*0.8)]
    test_data = mid_prices[int(len(mid_prices)*0.8):]

    return train_data, test_data

def predict_via_avaraging(train_data, test_data, all_mid_data, df):
    window_size = 100
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size,N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx,'Date']

        std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
    
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()

def moving_avarage(train_data, test_data, all_mid_data, df):
    window_size = 100
    N = train_data.size

    run_avg_predictions = []
    run_avg_x = []

    mse_errors = []

    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    decay = 0.5

    for pred_idx in range(1,N):

        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx,'Date']
        
        running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
        run_avg_x.append(date)

    print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
    plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()  

class LSTM(nn.Module):
    def __init__(self,input_dim,
                 hidden_dim,num_layers,
                 output_dim,
                 seq_size,
                 dropout=0):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self,x : torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm.forward(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out    

def create_dataset(dataset, lookback, lookahead):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback-lookahead):
        feature = dataset[i:i+lookback]
        target = dataset[i+lookahead:i+lookback+lookahead]
        X.append(feature)
        y.append(target)
    return torch.tensor(X.astype('float32')), torch.tensor(y.astype('float32'))

# def split_data_updated(X, y, precentage, lookback, lookahead):
#     """Split the dataset into training and test set batches
    
#     Args:
#         X: A torch tensor of input features (N, D)
#         y: A torch tensor of target (N, 1)
#         precentage: The precentage of training set
#     Output:
#         x_train: A torch tensor of input features for training set (B, T, D) 
#         y_train: A torch tensor of target for training set (T, 1)
#         x_test: A torch tensor of input features for test set (T, D)
#         y_test: A torch tensor of target for test set (T, 1)
#         T is the time steps, D is the dimension of features
#     """
#     T = lookback
#     D = X.shape[1]
    
#     for i in range(0, len(X), B):
#         if i == 0:
#             x_train = X[i:i+B]
#             y_train = y[i:i+B]
#         else:
#             x_train = torch.cat((x_train, X[i:i+B]), 0)
#             y_train = torch.cat((y_train, y[i:i+B]), 0)
    
    
    
#     return x_train, y_train, x_test, y_test

def split_train_test(dataset,percentage,lookback,lookahead):
    data_raw = dataset
    data = []
    
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
        
    data = np.array(data)
    val_size = int(np.round(percentage*data.shape[0]))
    train_size = int(np.round(percentage*val_size))
    
    x_train = data[:train_size, :-1]
    y_train = data[:train_size, -1]
    
    x_val = data[train_size:val_size, :-1]
    y_val = data[train_size:val_size, -1]
    
    x_test = data[val_size:, :-1]
    y_test = data[val_size:,-1]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def data_normalization(df_data):
    norm_data = df_data['Close'].values.reshape(-1)
    min_val = float(norm_data.min(axis=0))
    max_val = float(norm_data.max(axis=0))
    norm_data = 2*((norm_data-min_val)/(max_val-min_val))-1
    return norm_data, min_val, max_val

def re_scale_data(data, min_val, max_val):
    return (data+1)*(max_val-min_val)/2 + min_val

def train_model(model,
                optimiser,
                loss_function,
                x_train, y_train,
                x_val, y_val,
                num_epochs,
                look_ahead,
                print_every=100): 
    model.train()

    train_loss_curve = []
    val_loss_curve = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        # y_hat_train = torch.zeros((look_ahead, x_train.shape[0], 1))
        # x_train_with_look_ahead = torch.Tensor(x_train)
        # for i in range(look_ahead):
        #     if i == 0:
        #         y_hat_train[i] = model(x_train)
        #     else:
        #         x_train_with_look_ahead = torch.cat(
        #             (x_train_with_look_ahead[:, 1:, :],
        #              y_hat_train[i-1].reshape(-1, 1, 1)),
        #             0)
        #         y_hat_train[i] = model(x_train_with_look_ahead)
        y_hat_train = model(x_train)
        loss_train = loss_function(y_hat_train, y_train)
        train_loss_curve.append(loss_train.item())

        optimiser.zero_grad()
        loss_train.backward()
        optimiser.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_hat_val = model(x_val)
            loss_val = loss_function(y_hat_val, y_val)
            val_loss_curve.append(loss_val.item())

        if (epoch % print_every == 0):
            print('--- Epoch {}: Training Loss = {:.4f}, Validation Loss = {:.4f} ---'.format(epoch + 1, loss_train.item(), loss_val.item()))

    # Plot loss curves
    plt.plot(train_loss_curve, label='Training Loss')
    plt.plot(val_loss_curve, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    