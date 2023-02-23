from yahoo_fin.stock_info import get_data
import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dow_list = si.tickers_dow()


def get_stock_data_historical(list_of_tickers, start, end, interval, baseline_ticker):
	base_data = get_data(baseline_ticker, start_date=start, end_date=end, index_as_date = True, interval=interval)[['close']]
	base_data.rename(columns = {'close':baseline_ticker}, inplace = True)
	for ticker in list_of_tickers:
		try:
			temp = get_data(ticker, start_date=start, end_date=end, index_as_date = True, interval=interval)[['close']]
			temp.rename(columns = {'close':ticker}, inplace = True)
			base_data = base_data.merge(temp,how='left',left_index=True, right_index=True)
		except Exception:
			pass
	return base_data



dow_historical_data = get_stock_data_historical(dow_list,"12/04/2017","12/04/2022", "1wk", "spy")