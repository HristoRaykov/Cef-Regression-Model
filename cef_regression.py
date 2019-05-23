from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from enum import Enum
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

CEF_DATA_SOURCES = {"ADX": ["data/adx_hist_prices.csv", "data/adx_hist_navs.csv"],
                    "CII": ["data/cii_hist_prices.csv", "data/cii_hist_navs.csv"],
                    "EOS": ["data/eos_hist_prices.csv", "data/eos_hist_navs.csv"]}

DATA_FILE_POSTFIX = "_data.csv"
Z_SCORE_POSTFIX = " Z-score"
RETURN_PREFIX = "Return on "

DATA_PATH_PREFIX = "data/"
DATE_COL_NAME = "Date"
PRICE_COL_NAME = "Price"
PRICE_RETURNS_COL_NAME = RETURN_PREFIX + PRICE_COL_NAME
NAV_COL_NAME = "NAV"
NAV_RETURNS_COL_NAME = RETURN_PREFIX + NAV_COL_NAME
PREM_DISC_COL_NAME = "Prem/Disc"
PREM_DISC_ZSCORE_COL_NAME = PREM_DISC_COL_NAME + Z_SCORE_POSTFIX
RESIDUALS_COL_NAME = "Res"
RESIDUAL_ZSCORE_COL_NAME = RESIDUALS_COL_NAME + str(Z_SCORE_POSTFIX)
ACTION_COL_NAME = "Action"
PROFIT_DELTA_COL_NAME = "P&L Delta $"
PROFIT_DELTA_PERC_COL_NAME = "P&L Delta %"
CUMULATIVE_PROFIT_COL_NAME = "Cum P&L $"
HOLDING_PERIOD_COL_NAME = "HP (days)"
X_COL_NAME = "Nav Returns"
Y_COL_NAME = "Actual Price Returns"
Y_PREDICTED_COL_NAME = "Predicted Price Returns"

CEF_TICKER = "ADX"
REGRESSOR_COL_NAME = NAV_RETURNS_COL_NAME
TRAIN_DATA_RATIO = 0.75
PERIOD_3MONTHS = relativedelta(months=3)
PERIOD_6MONTHS = relativedelta(months=6)
PERIOD_1YEAR = relativedelta(years=1)
PERIOD_2YEAR = relativedelta(years=2)
AVERAGES_CALC_PERIOD = PERIOD_3MONTHS
ANALYSIS_DATA_PERIOD = PERIOD_1YEAR


class TradeAction(Enum):
	BUY_LONG = "Buy Long"
	COVER_LONG = "Cover Long"
	SELL_SHORT = "Sell Short"
	COVER_SHORT = "Cover Short"


class TradePosition(Enum):
	LONG = "Long"
	SHORT = "Short"
	NO_POSITION = "No position"


def read_raw_cef_data(cef_symbol):
	"""
	Read Closed-End Fund price and Net Asset Value (NAV) quotes from csv file format.
	Cef tickers and file names are taken from user defined dict CEF_DATA_SOURCES
	:param cef_symbol: Ticker of the financial asset.
	:return: pandas DataFrame with cols: [DATE_COL_NAME, PRICE_COL_NAME, NAV_COL_NAME]
	"""
	cef = pd.read_csv(CEF_DATA_SOURCES[cef_symbol][0])
	cef = cef[["timestamp", "close"]]
	cef.columns = [DATE_COL_NAME, PRICE_COL_NAME]
	cef[NAV_COL_NAME] = pd.read_csv(CEF_DATA_SOURCES[cef_symbol][1])["close"]
	cef[DATE_COL_NAME] = pd.to_datetime(cef[DATE_COL_NAME])
	cef = cef.sort_values([DATE_COL_NAME])
	return cef


def calculate_zscore(df, col_name, period_start_date, period_end_date):
	"""
	Calculating Z-score of the value in the cell [df[DATE_COL_NAME]==period_end_date, col_name]
	from period_start_date to period_end_date.
	DataFrame should contain column DATE_COL_NAME and col_name.
	
	:return: DataFrame with calculated z-score for 'period_end_date'
	"""
	data = df.loc[
		(df[DATE_COL_NAME] >= period_start_date) & (df[DATE_COL_NAME] <= period_end_date), col_name]
	curr_value = df.loc[df[DATE_COL_NAME] == period_end_date, col_name].values[0]
	average_value = data.mean()
	std = data.std()
	zscore = (curr_value - average_value) / std
	df.loc[df[DATE_COL_NAME] == period_end_date, col_name + Z_SCORE_POSTFIX] = zscore
	return df


def calculate_return(df, col_name, period_start_date, period_end_date):
	"""
	Calculating holding period return from period_start_date to period_end_date.
	DataFrame should contain column DATE_COL_NAME and col_name.
	
	:return: DataFrame with calculated return for 'period_end_date'
	"""
	base_value = df.loc[df[DATE_COL_NAME] == period_start_date, col_name].values[0]
	curr_value = df.loc[df[DATE_COL_NAME] == period_end_date, col_name].values[0]
	price_return = (curr_value - base_value) / base_value * 100
	df.loc[df[DATE_COL_NAME] == period_end_date, RETURN_PREFIX + col_name] = price_return
	return df


def find_valid_period_start_date(dates, date, period):
	"""
	Select an existing period begin date in ordered list of dates
	"""
	
	period_start_date = date - period
	period_dates = dates[dates >= period_start_date]
	first_date = period_dates.iloc[0]
	return first_date


def calculate_trailing_residual_zscores(df_input, regressor_col_name, simulation_begin_date, calc_period):
	"""
	Calculating trailing calc_period z-scores for residuals = y - y_predicted
	from simulation_begin_date to last date, the regressand is always Price Returns.
	
	:return: DataFrame with calculated residual z-score column base on trailing history calc_period
	"""
	df = df_input.copy()
	all_dates = df[DATE_COL_NAME]
	dates = df.loc[df[DATE_COL_NAME] >= simulation_begin_date, DATE_COL_NAME].reset_index(drop=True)
	for date in dates:
		period_start_date = find_valid_period_start_date(all_dates, date, calc_period)
		data = df[(df[DATE_COL_NAME] >= period_start_date) & (df[DATE_COL_NAME] <= date)].reset_index(drop=True)
		x = data[regressor_col_name].values.reshape(-1, 1)
		y = np.array(data[PRICE_RETURNS_COL_NAME].values)
		
		regress_model = LinearRegression().fit(x, y)
		y_predicted = regress_model.predict(x)
		residuals = y - y_predicted
		
		curr_val = residuals[len(residuals) - 1]
		average_value = residuals.mean()
		std = residuals.std()
		zscore = (curr_val - average_value) / std
		df.loc[df[DATE_COL_NAME] == date, RESIDUAL_ZSCORE_COL_NAME] = zscore
	
	df = df.loc[df[DATE_COL_NAME] >= simulation_begin_date].reset_index(drop=True)
	return df


def calculate_factors(cef, start_date, period):
	"""
	Calculating price returns, nav returns and premium/discount z-scores
	based on history 'period' for analyzing Closed End Fund performance.

	:return: DataFrame with calculated factors
	"""
	cef = cef[(cef[DATE_COL_NAME] >= start_date - period)]
	all_dates = cef[DATE_COL_NAME].reset_index(drop=True)
	cef[PREM_DISC_COL_NAME] = (cef[PRICE_COL_NAME] - cef[NAV_COL_NAME]) / cef[NAV_COL_NAME] * 100
	
	dates = cef.loc[cef[DATE_COL_NAME] >= start_date, DATE_COL_NAME].reset_index(drop=True)
	for date in dates:
		period_start_date = find_valid_period_start_date(all_dates, date, period)
		cef = calculate_return(cef, PRICE_COL_NAME, period_start_date, date)
		cef = calculate_return(cef, NAV_COL_NAME, period_start_date, date)
		cef = calculate_zscore(cef, PREM_DISC_COL_NAME, period_start_date, date)
	return cef


def calculate_cef_data(symbol, analysis_data_period, calc_period):
	"""
	Processing raw input Cef data which may take some time. Calculating factors for conducting analysis
	and saving data to new csv file. This method should be called only if we made changes to TRAIN_DATA_RATIO,
	AVERAGES_CALC_PERIOD, START_DATE etc.

	"""
	print("Processing data ... Please wait ... for 'Data processing finished!' indication below!")
	df = read_raw_cef_data(symbol)
	dates = df[DATE_COL_NAME].reset_index(drop=True)
	end_date = str(dates.values[-1]).split("T")[0]
	end_date = datetime.strptime(end_date, "%Y-%m-%d")
	start_date = find_valid_period_start_date(dates, end_date, analysis_data_period)
	df = calculate_factors(df, start_date, calc_period).reset_index(drop=True)
	df = df.loc[df[DATE_COL_NAME] >= start_date].reset_index(drop=True)
	file_name = symbol.lower() + DATA_FILE_POSTFIX
	path = DATA_PATH_PREFIX + file_name
	df.to_csv(path)
	print("Data processing finished!")


def split_train_test_data(cef, split_data_coefficient):
	"""
	Split explored data time series into two periods for training and testing based on coefficient.
	"""
	train_data_end_index = int((len(cef) - 1) * split_data_coefficient)
	cef_train_data = cef.loc[cef.index <= train_data_end_index]
	cef_test_data = cef.loc[cef.index > train_data_end_index].reset_index(drop=True)
	return cef_train_data, cef_test_data


def analyze_regression(cef_ticker, cef_train_data, cef_test_data, regressor_col_name, regressand_col_name):
	"""
	Applying regression on train data and test the model on test data.
	"""
	regressor_train_data = cef_train_data[regressor_col_name].values.reshape(-1, 1)
	regressand_train_data = np.array(cef_train_data[regressand_col_name].values)
	
	regressor_test_data = cef_test_data[regressor_col_name].values.reshape(-1, 1)
	regressand_actual_test_data = np.array(cef_test_data[regressand_col_name].values)
	
	regress_model = LinearRegression().fit(regressor_train_data, regressand_train_data)
	intercept = regress_model.intercept_
	coef = regress_model.coef_[0]
	
	r_sq_x_y = regress_model.score(regressor_train_data, regressand_train_data)
	regressand_predicted_test_data = regress_model.predict(regressor_test_data)
	residuals = regressand_actual_test_data - regressand_predicted_test_data
	dates = cef_test_data[DATE_COL_NAME]
	
	corr_train_data_x_y = cef_train_data[regressor_col_name].corr(cef_train_data[regressand_col_name])
	corr_test_predicted_actual_y = np.corrcoef(regressand_predicted_test_data, regressand_actual_test_data)[0, 1]
	
	regress_data = pd.DataFrame.from_dict({DATE_COL_NAME: dates.values,
	                                       RESIDUALS_COL_NAME: residuals,
	                                       X_COL_NAME: list(regressor_test_data),
	                                       Y_COL_NAME: regressand_actual_test_data,
	                                       Y_PREDICTED_COL_NAME: regressand_predicted_test_data})
	regress_statistics = {"cef_ticker": cef_ticker,
	                      "regressor": regressor_col_name,
	                      "intercept": intercept,
	                      "coef": coef,
	                      "R-square": r_sq_x_y,
	                      "Corr_x_y": corr_train_data_x_y,
	                      "Corr_pred_actual_y": corr_test_predicted_actual_y}
	
	return regress_data, regress_statistics


def read_processed_cef_data(cef_ticker):
	"""
	Reading processed Cef data from file as DataFrame.
	"""
	path = DATA_PATH_PREFIX + cef_ticker.lower() + DATA_FILE_POSTFIX
	cef = pd.read_csv(path, index_col=0)
	cef[DATE_COL_NAME] = pd.to_datetime(cef[DATE_COL_NAME])
	cef = cef[[DATE_COL_NAME, PRICE_COL_NAME, PRICE_RETURNS_COL_NAME, NAV_RETURNS_COL_NAME, PREM_DISC_ZSCORE_COL_NAME]]
	return cef


def run_residual_trade_simulation(trade_simul_data, zscore_buy_long=-1.5, zscore_cover_long=-0.5, zscore_sell_short=1.5,
                                  zscore_cover_short=0.5):
	"""
	Running trade simulation on the given time series based on residual z-score.
	Simulation takes long and short positions. One position at a time.
	:return: trades DataFrame and time series DataFrame of profits and losses.
	"""
	data = trade_simul_data[[DATE_COL_NAME, PRICE_COL_NAME, RESIDUAL_ZSCORE_COL_NAME]]
	trades = pd.DataFrame(
		columns=[DATE_COL_NAME, PRICE_COL_NAME, RESIDUAL_ZSCORE_COL_NAME, ACTION_COL_NAME, HOLDING_PERIOD_COL_NAME,
		         PROFIT_DELTA_COL_NAME, CUMULATIVE_PROFIT_COL_NAME])
	dates = trade_simul_data[DATE_COL_NAME]
	continuos_profits = pd.DataFrame(
		columns=[DATE_COL_NAME, PRICE_COL_NAME, PROFIT_DELTA_COL_NAME, CUMULATIVE_PROFIT_COL_NAME])
	continuos_profits[DATE_COL_NAME] = dates
	continuos_profits[PRICE_COL_NAME] = trade_simul_data[PRICE_COL_NAME]
	
	trade_position = TradePosition.NO_POSITION
	cum_realized_profit = 0
	cum_continous_profit = 0
	for i in data.index.values:
		row = data.iloc[i]
		curr_date = row[DATE_COL_NAME]
		curr_price = row[PRICE_COL_NAME]
		residual_zscore = row[RESIDUAL_ZSCORE_COL_NAME]
		daily_profit_delta = 0
		
		if trade_position == TradePosition.NO_POSITION:
			holding_period = 0
			realized_profit_delta = 0
			if residual_zscore <= zscore_buy_long:
				action = TradeAction.BUY_LONG
				trade_row = list(
					np.concatenate([row.values,
					                [action, holding_period, realized_profit_delta,
					                 cum_realized_profit]]))
				trades = append_row(trades, trade_row)
				trade_position = TradePosition.LONG
			elif residual_zscore >= zscore_sell_short:
				action = TradeAction.SELL_SHORT
				trade_row = list(
					np.concatenate([row.values,
					                [action, holding_period, realized_profit_delta,
					                 cum_realized_profit]]))
				trades = append_row(trades, trade_row)
				trade_position = TradePosition.SHORT
		else:
			previous_day_price = data.iloc[i - 1, 1]
			entry_date = trades.iloc[-1, 0]
			holding_period = (curr_date - entry_date).days
			entry_price = trades.iloc[-1, 1]
			if trade_position == TradePosition.LONG:
				daily_profit_delta = curr_price - previous_day_price
				if residual_zscore >= zscore_cover_long:
					action = TradeAction.COVER_LONG
					realized_profit_delta = curr_price - entry_price
					cum_realized_profit += realized_profit_delta
					trade_row = list(
						np.concatenate([row.values,
						                [action, holding_period, realized_profit_delta,
						                 cum_realized_profit]]))
					trades = append_row(trades, trade_row)
					trade_position = TradePosition.NO_POSITION
			elif trade_position == TradePosition.SHORT:
				daily_profit_delta = previous_day_price - curr_price
				if residual_zscore <= zscore_cover_short:
					action = TradeAction.COVER_SHORT
					realized_profit_delta = entry_price - curr_price
					cum_realized_profit += realized_profit_delta
					trade_row = list(
						np.concatenate([row.values,
						                [action, holding_period, realized_profit_delta,
						                 cum_realized_profit]]))
					trades = append_row(trades, trade_row)
					trade_position = TradePosition.NO_POSITION
			cum_continous_profit += daily_profit_delta
		
		continuos_profits.iloc[i, 2] = daily_profit_delta
		continuos_profits.iloc[i, 3] = cum_continous_profit
	
	return trades, continuos_profits


def append_row(trades, trade_row):
	trades = trades.append({DATE_COL_NAME: trade_row[0],
	                        PRICE_COL_NAME: trade_row[1],
	                        RESIDUAL_ZSCORE_COL_NAME: trade_row[2],
	                        ACTION_COL_NAME: trade_row[3].name,
	                        HOLDING_PERIOD_COL_NAME: trade_row[4],
	                        PROFIT_DELTA_COL_NAME: trade_row[5],
	                        CUMULATIVE_PROFIT_COL_NAME: trade_row[6]}, ignore_index=True)
	return trades


def plot_trade_simulation(cef_ticker, trades, continuous_profits):
	dates = continuous_profits[DATE_COL_NAME]
	prices = continuous_profits[PRICE_COL_NAME]
	profit_deltas = continuous_profits[PROFIT_DELTA_COL_NAME]
	cum_profit = continuous_profits[CUMULATIVE_PROFIT_COL_NAME]
	buys = trades.loc[
		(trades[ACTION_COL_NAME] == TradeAction.BUY_LONG.name) | (
				trades[ACTION_COL_NAME] == TradeAction.COVER_SHORT.name), [DATE_COL_NAME, PRICE_COL_NAME]]
	sells = trades.loc[
		(trades[ACTION_COL_NAME] == TradeAction.SELL_SHORT.name) | (
				trades[ACTION_COL_NAME] == TradeAction.COVER_LONG.name), [DATE_COL_NAME, PRICE_COL_NAME]]
	
	fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 4))
	
	ax1.plot(dates, prices, alpha=0.6)
	ax1.scatter(buys[DATE_COL_NAME], buys[PRICE_COL_NAME], marker="^", c="g", label="Buy Prices")
	ax1.scatter(sells[DATE_COL_NAME], sells[PRICE_COL_NAME], marker="v", c="r", label="Sell Prices")
	ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
	ax1.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
	ax1.set_xlabel("Dates", fontsize=10)
	ax1.set_ylabel("Prices", fontsize=10)
	ax1.tick_params(axis='both', which='both', labelsize=8, rotation=30)
	ax1.set_title("Trade Simulation", fontsize=12)
	ax1.legend(loc=4, prop={"size": 8})
	
	ax2.bar(dates, profit_deltas)
	ax2.axhline(y=0, linewidth=1, color='k')
	ax2.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
	ax2.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
	ax2.set_xlabel("Date", fontsize=10)
	ax2.set_ylabel("Profit Deltas", fontsize=10)
	ax2.tick_params(axis='both', which='both', labelsize=8, rotation=30)
	ax2.set_title("Profit Daily Changes", fontsize=12)
	
	ax3.plot(dates, cum_profit)
	ax3.axhline(y=0, linewidth=1, color='k')
	ax3.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
	ax3.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
	ax3.set_xlabel("Dates", fontsize=10)
	ax3.set_ylabel("Profit/Loss In $", fontsize=10)
	ax3.tick_params(axis='both', which='both', labelsize=8, rotation=30)
	ax3.set_title("Profit/Loss Dynamics", fontsize=12)
	ax3.legend(loc=3, prop={"size": 8})
	
	fig.suptitle(cef_ticker)
	plt.subplots_adjust(wspace=0.25)
	plt.show()


def plot_regress_result(regress_data, regress_statistics):
	dates = regress_data[DATE_COL_NAME]
	residuals = regress_data[RESIDUALS_COL_NAME]
	x = regress_data[X_COL_NAME]
	y_predicted = regress_data[Y_PREDICTED_COL_NAME]
	y_actual = regress_data[Y_COL_NAME]
	cef_ticker = regress_statistics["cef_ticker"]
	regressor = regress_statistics["regressor"]
	intercept = regress_statistics["intercept"]
	coef = regress_statistics["coef"]
	corr_x_y = regress_statistics["Corr_x_y"]
	r_sq = regress_statistics["R-square"]
	corr_pred_actual = regress_statistics["Corr_pred_actual_y"]
	text_ax1 = "y = {0:.3f} + {1:.3f}.x\n" \
	           "corr = {2:.3f}\n" \
	           "r_sq = {3:.3f}".format(intercept, coef, corr_x_y, r_sq)
	text_ax2 = "corr_pred_vs_actual = {0:.3f}".format(corr_pred_actual)
	
	fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(13, 10))
	
	ax1.plot(x, y_predicted, label="Predicted Price Returns", c="b")
	ax1.scatter(x, y_actual, c="y", s=14, label="Actual Pice Returns")
	ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))
	ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))
	ax1.set_xlabel(regressor, fontsize=10)
	ax1.set_ylabel("Price Returns", fontsize=10)
	ax1.tick_params(axis='both', which='both', labelsize=8, rotation=30)
	ax1.text(0.05, 0.85, text_ax1, color="b", ha='left', va='center', transform=ax1.transAxes, fontsize=12)
	ax1.set_title("{0}-Price Returns Regression Model".format(regressor), fontsize=12)
	ax1.legend(loc=4, prop={"size": 8})
	
	ax2.plot(dates, y_predicted, label="Predicted Price Returns", c="b")
	ax2.plot(dates, y_actual, label="Actual Pice Returns", c="y")
	ax2.axhline(y=0, linewidth=1, color='k')
	ax2.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
	ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))
	ax2.set_xlabel("Date", fontsize=10)
	ax2.set_ylabel("Price Returns", fontsize=10)
	ax2.tick_params(axis='both', which='both', labelsize=8, rotation=30)
	ax2.text(0.05, 0.95, text_ax2, color="b", ha='left', va='center', transform=ax2.transAxes, fontsize=11)
	ax2.set_title("Predicted vs Actual Price Returns", fontsize=12)
	ax2.legend(loc=4, prop={"size": 8})
	
	ax3.bar(dates, residuals)
	ax3.axhline(y=0, linewidth=1, color='k')
	ax3.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
	ax3.yaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))
	ax3.set_xlabel("Date", fontsize=10)
	ax3.set_ylabel("Residuals", fontsize=10)
	ax3.tick_params(axis='both', which='both', labelsize=8, rotation=30)
	ax3.set_title("Predicted - Actual Price Returns (residuals)", fontsize=12)
	
	ax4.hist(residuals, bins="fd")
	ax4.xaxis.set_major_formatter(FormatStrFormatter("%.2f%%"))
	ax4.set_xlabel("Residuals", fontsize=10)
	ax4.set_ylabel("Count", fontsize=10)
	ax4.tick_params(axis='both', which='both', labelsize=8, rotation=30)
	ax4.set_title("Residuals  Distribution", fontsize=12)
	
	fig.suptitle(cef_ticker)
	plt.subplots_adjust(hspace=0.3, top=0.93)
	plt.show()


def main():
	# calculate_cef_data(CEF_TICKER, ANALYSIS_DATA_PERIOD, AVERAGES_CALC_PERIOD)
	cef = read_processed_cef_data(CEF_TICKER)
	cef_train_data, cef_test_data = split_train_test_data(cef, TRAIN_DATA_RATIO)

	regress_data, regress_statistics = analyze_regression(CEF_TICKER, cef_train_data, cef_test_data, REGRESSOR_COL_NAME,
	                                                      PRICE_RETURNS_COL_NAME)
	plot_regress_result(regress_data, regress_statistics)

	simulation_start_date = cef_test_data[DATE_COL_NAME].values[0]
	cef_simul_data = calculate_trailing_residual_zscores(cef, REGRESSOR_COL_NAME, simulation_start_date,
	                                                     AVERAGES_CALC_PERIOD)

	trades, continuos_profits = run_residual_trade_simulation(cef_simul_data)
	plot_trade_simulation(CEF_TICKER, trades, continuos_profits)

	print(trades)


if __name__ == "__main__":
	main()
