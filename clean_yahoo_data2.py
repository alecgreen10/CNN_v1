import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


TRAIN_TEST_CUTOFF = '01-04-2022'
TRAIN_VALID_RATIO = 0.75



dow_historical_data = pd.read_csv('train_1.csv', index_col=[0])


def simple_stock_data(historical_data, train_cutoff, train_ratio):
	## |date|aapl px|amgn px|
	##
	new_df = historical_data[['AAPL','AMGN']].copy()
	cols = new_df.columns
	new_df["Target"] = (new_df["AAPL"].pct_change(4).shift(-4) > 0).astype(int)
	new_df.dropna(inplace=True)
	new_df.index = pd.to_datetime(new_df.index)
	index = new_df.index[new_df.index < train_cutoff]
	index = index[:int(len(index) * train_ratio)]
	scaler = StandardScaler().fit(new_df.loc[index, cols])
	new_df[cols] = scaler.transform(new_df[cols])



def single_stock_data(historical_data, train_cutoff, train_ratio, stock, holding_period):
	## | date | aapl px | spy px | aapl 2wk | aapl 3wk ... aapl 12wk | aapl stdev 1wk returns | target |
	new_df = historical_data[['AMGN','spy']].copy()
	# new_df["1wk return"] = new_df["AMGN"].pct_change(1)
	# new_df["2wk return"] = new_df["AMGN"].pct_change(2)
	# new_df["3wk return"] = new_df["AMGN"].pct_change(3)
	# new_df["4wk return"] = new_df["AMGN"].pct_change(4)
	# new_df["5wk return"] = new_df["AMGN"].pct_change(5)
	# new_df["6wk return"] = new_df["AMGN"].pct_change(6)
	# new_df["7wk return"] = new_df["AMGN"].pct_change(7)
	# new_df["8wk return"] = new_df["AMGN"].pct_change(8)
	# new_df["9wk return"] = new_df["AMGN"].pct_change(9)
	# new_df["10wk return"] = new_df["AMGN"].pct_change(10)
	# new_df["11wk return"] = new_df["AMGN"].pct_change(11)
	new_df["12wk return"] = new_df["AMGN"].pct_change(12)
	# new_df["5wk return spy"] = new_df["spy"].pct_change(5)
	# new_df["6wk return spy"] = new_df["spy"].pct_change(6)
	# new_df["7wk return spy"] = new_df["spy"].pct_change(7)
	# new_df["8wk return spy"] = new_df["spy"].pct_change(8)
	# new_df["9wk return spy"] = new_df["spy"].pct_change(9)
	new_df["12wk return spy"] = new_df["spy"].pct_change(12)
	cols = new_df.columns
	new_df["Target1"] = new_df["AMGN"].pct_change(holding_period).shift(-holding_period)
	new_df["Target2"] = new_df["spy"].pct_change(holding_period).shift(-holding_period)  #4 week holding period
	new_df["Target"] = new_df["Target1"] - new_df["Target2"]
	new_df = new_df.drop(columns=['Target1', 'Target2'])
	new_df.dropna(inplace=True)
	new_df["Target"] = (new_df["Target"] > 0.01).astype(int) #4 week holding period
	new_df.index = pd.to_datetime(new_df.index)
	index = new_df.index[new_df.index < train_cutoff]
	index = index[:int(len(index) * train_ratio)]
	scaler = StandardScaler().fit(new_df.loc[index, cols])
	new_df[cols] = scaler.transform(new_df[cols])
	return new_df


def generate_class_weights(df, train_cutoff):
	# df.index = pd.to_datetime(df.index)
	weights = {}
	df = df.loc[:train_cutoff]
	num_samples = len(df.index)
	num_1 = df['Target'].sum()
	num_0 = num_samples - num_1
	w_1 = num_samples / (2 * num_1)
	w_0 = num_samples / (2 * num_0)
	weights[0] = w_0
	weights[1] = w_1
	return weights


x = single_stock_data(dow_historical_data, TRAIN_TEST_CUTOFF, TRAIN_VALID_RATIO, 'AMGN', 4)
class_weight = generate_class_weights(x,TRAIN_TEST_CUTOFF)


















