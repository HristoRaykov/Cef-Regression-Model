from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from enum import Enum
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns

BENCHMARK_INDEX_SOURCE = {"SPX": "spx_hist_prices.csv"}
CEF_DATA_SOURCES = {"ADX": ["adx_hist_prices.csv", "adx_hist_navs.csv"],
                    "CII": ["cii_hist_prices.csv", "cii_hist_navs.csv"],
                    "EOS": ["eos_hist_prices.csv", "eos_hist_navs.csv"]}

CEF_TICKER = "ADX"

DATA_FILE_POSTFIX = "_data.csv"
Z_SCORE_POSTFIX = " Z-score"
RETURN_PREFIX = "Return on "

DATE_COL_NAME = "Date"
PRICE_COL_NAME = "Price"
PRICE_RETURNS_COL_NAME = RETURN_PREFIX + PRICE_COL_NAME
NAV_COL_NAME = "NAV"
NAV_RETURNS_COL_NAME = RETURN_PREFIX + NAV_COL_NAME
PREM_DISC_COL_NAME = "Prem/Disc"
PREM_DISC_ZSCORE_COL_NAME = PREM_DISC_COL_NAME + Z_SCORE_POSTFIX
RESIDUALS_COL_NAME = "Residual"
RESIDUAL_ZSCORE_COL_NAME = RESIDUALS_COL_NAME + str(Z_SCORE_POSTFIX)
ACTION_COL_NAME = "Action"
PROFIT_DELTA_COL_NAME = "P&L Delta $"
PROFIT_DELTA_PERC_COL_NAME = "P&L Delta %"
CUMULATIVE_PROFIT_COL_NAME = "Cumulative P&L $"
HOLDING_PERIOD_COL_NAME = "Holding Period (days)"
Y_COL_NAME = "Actual Returns"
Y_PREDICTED_COL_NAME = "Predicted Returns"

TRAIN_DATA_RATIO = 0.75
AVERAGES_CALC_PERIOD = relativedelta(months=3)
START_DATE = datetime(2018, 5, 13)


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
	cef = pd.read_csv(CEF_DATA_SOURCES[cef_symbol][0])
	cef = cef[["timestamp", "close"]]
	cef.columns = [DATE_COL_NAME, PRICE_COL_NAME]
	cef[NAV_COL_NAME] = pd.read_csv(CEF_DATA_SOURCES[cef_symbol][1])["close"]
	cef[DATE_COL_NAME] = pd.to_datetime(cef[DATE_COL_NAME])
	cef = cef.sort_values([DATE_COL_NAME])
	return cef


def read_index_data(index_symbol):
	index = pd.read_csv(BENCHMARK_INDEX_SOURCE[index_symbol])
	index = index[["timestamp", "close"]]
	index.columns = [DATE_COL_NAME, PRICE_COL_NAME]
	index[DATE_COL_NAME] = pd.to_datetime(index[DATE_COL_NAME])
	index = index.sort_values([DATE_COL_NAME])
	return index


def calculate_zscore(df, col_name, period_start_date, period_end_date):
	data = df.loc[
		(df[DATE_COL_NAME] >= period_start_date) & (df[DATE_COL_NAME] <= period_end_date), col_name]
	curr_value = df.loc[df[DATE_COL_NAME] == period_end_date, col_name].values[0]
	average_value = data.mean()
	std = data.std()
	zscore = (curr_value - average_value) / std
	# cef.loc[cef["date"] == period_end_date, "nav mean"] = average_prem_disc
	# cef.loc[cef["date"] == period_end_date, "nav std"] = std_prem_disc
	df.loc[df[DATE_COL_NAME] == period_end_date, col_name + Z_SCORE_POSTFIX] = zscore
	return df


def calculate_return(df, col_name, period_start_date, period_end_date):
	base_value = df.loc[df[DATE_COL_NAME] == period_start_date, col_name].values[0]
	curr_value = df.loc[df[DATE_COL_NAME] == period_end_date, col_name].values[0]
	price_return = (curr_value - base_value) / base_value * 100
	df.loc[df[DATE_COL_NAME] == period_end_date, RETURN_PREFIX + col_name] = price_return
	return df


def find_valid_period_start_date(dates, date, period):
	period_start_date = date - period
	period_dates = dates[dates >= period_start_date]
	first_date = period_dates.iloc[0]
	return first_date


def calculate_residual_zscores(df, simulation_begin_date, calc_period):
	all_dates = df[DATE_COL_NAME]
	dates = df.loc[df[DATE_COL_NAME] >= simulation_begin_date, DATE_COL_NAME]
	for date in dates:
		period_start_date = find_valid_period_start_date(all_dates, date, calc_period)
		data = df[(df[DATE_COL_NAME] >= period_start_date) & (df[DATE_COL_NAME] <= date)]
		x = data[PRICE_RETURNS_COL_NAME].values.reshape(-1, 1)
		y = np.array(data[NAV_RETURNS_COL_NAME].values)
		
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
	cef = cef[(cef[DATE_COL_NAME] >= start_date - period)]
	all_dates = cef[DATE_COL_NAME]
	cef[PREM_DISC_COL_NAME] = (cef[PRICE_COL_NAME] - cef[NAV_COL_NAME]) / cef[NAV_COL_NAME] * 100
	
	dates = cef.loc[cef[DATE_COL_NAME] >= start_date, DATE_COL_NAME]
	for date in dates:
		period_start_date = find_valid_period_start_date(all_dates, date, period)
		cef = calculate_return(cef, PRICE_COL_NAME, period_start_date, date)
		cef = calculate_return(cef, NAV_COL_NAME, period_start_date, date)
		cef = calculate_zscore(cef, PREM_DISC_COL_NAME, period_start_date, date)
		print("Processing data for " + str(date))
	return cef


def calculate_cef_data(symbol, start_date, calc_period):
	df = read_raw_cef_data(symbol)
	df = calculate_factors(df, start_date, calc_period).reset_index(drop=True)
	df = df.loc[df[DATE_COL_NAME] >= start_date].reset_index(drop=True)
	file_name = symbol.lower() + DATA_FILE_POSTFIX
	df.to_csv(file_name)


def split_train_test_data(cef, split_data_coefficient):
	train_data_end_index = int((len(cef) - 1) * split_data_coefficient)
	cef_train_data = cef.loc[cef.index <= train_data_end_index]
	cef_test_data = cef.loc[cef.index > train_data_end_index].reset_index(drop=True)
	return cef_train_data, cef_test_data


def analyze_regression(cef_train_data, cef_test_data, regressor_col_name, regressand_col_name):
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
	                                       Y_COL_NAME: regressand_actual_test_data,
	                                       Y_PREDICTED_COL_NAME: regressand_predicted_test_data})
	regress_statistics = {"intercept": intercept,
	                      "coef": coef,
	                      "R-square": r_sq_x_y,
	                      "Corr_x_y": corr_train_data_x_y,
	                      "Corr_pred_actual_y": corr_test_predicted_actual_y}
	return regress_data, regress_statistics


# plt.scatter(regressand_predicted_test_data, regressand_actual_test_data)
# plt.show()
# plt.bar(dates, residuals)
# plt.xticks(rotation=90)
# plt.show()
# plt.plot(dates, regressand_predicted_test_data, label="Predicted")
# plt.plot(dates, regressand_actual_test_data, label="Actual")
# plt.legend()
# plt.xticks(rotation=90)
# plt.show()
#
# print("y = {0:.3f} + {1:.3f}*x".format(regress_model.intercept_, regress_model.coef_[0]))
# print("corr_train_data = {0:.3f}".format(corr_train_data))
# print("R-square = {0:.3f}".format(r_sq))
# print("corr_predicted_actual = {0:.3f}".format(corr_predicted_actual))


def read_processed_cef_data(cef_ticker):
	cef = pd.read_csv(cef_ticker.lower() + DATA_FILE_POSTFIX, index_col=0)
	cef[DATE_COL_NAME] = pd.to_datetime(cef[DATE_COL_NAME])
	cef = cef[[DATE_COL_NAME, PRICE_COL_NAME, PRICE_RETURNS_COL_NAME, NAV_RETURNS_COL_NAME, PREM_DISC_ZSCORE_COL_NAME]]
	return cef


def run_trade_simulation(trade_simul_data, zscore_buy_long=-1, zscore_cover_long=0, zscore_sell_short=1,
                         zscore_cover_short=0):
	data = trade_simul_data[[DATE_COL_NAME, PRICE_COL_NAME, RESIDUAL_ZSCORE_COL_NAME]]
	trades = pd.DataFrame(
		columns=[DATE_COL_NAME, PRICE_COL_NAME, RESIDUAL_ZSCORE_COL_NAME, ACTION_COL_NAME, HOLDING_PERIOD_COL_NAME,
		         PROFIT_DELTA_COL_NAME, PROFIT_DELTA_PERC_COL_NAME, CUMULATIVE_PROFIT_COL_NAME])
	dates = trade_simul_data[DATE_COL_NAME]
	continuos_profits = pd.DataFrame(columns=[DATE_COL_NAME, PROFIT_DELTA_COL_NAME, CUMULATIVE_PROFIT_COL_NAME])
	continuos_profits[DATE_COL_NAME] = dates
	
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
			realized_profit_delta_perc = 0
			if residual_zscore <= zscore_buy_long:
				action = TradeAction.BUY_LONG
				trade_row = list(
					np.concatenate([row.values,
					                [action, holding_period, realized_profit_delta, realized_profit_delta_perc,
					                 cum_realized_profit]]))
				trades = append_row(trades, trade_row)
				trade_position = TradePosition.LONG
			elif residual_zscore >= zscore_sell_short:
				action = TradeAction.SELL_SHORT
				trade_row = list(
					np.concatenate([row.values,
					                [action, holding_period, realized_profit_delta, realized_profit_delta_perc,
					                 cum_realized_profit]]))
				trades = append_row(trades, trade_row)
				trade_position = TradePosition.LONG
		else:
			previous_day_price = data.iloc[i - 1, 1]
			entry_date = trades.iloc[-1, 0]
			holding_period = entry_date - curr_date
			entry_price = trades.iloc[-1, 1]
			if trade_position == TradePosition.LONG:
				daily_profit_delta = curr_price - previous_day_price
				if residual_zscore >= zscore_cover_long:
					action = TradeAction.COVER_LONG
					realized_profit_delta = curr_price - entry_price
					realized_profit_delta_perc = realized_profit_delta / entry_price * 100
					cum_realized_profit += realized_profit_delta
					trade_row = list(
						np.concatenate([row.values,
						                [action, holding_period, realized_profit_delta, realized_profit_delta_perc,
						                 cum_realized_profit]]))
					trades = append_row(trades, trade_row)
					trade_position = TradePosition.NO_POSITION
			elif trade_position == TradePosition.SHORT:
				daily_profit_delta = previous_day_price - curr_price
				if residual_zscore <= zscore_cover_short:
					action = TradeAction.COVER_SHORT
					realized_profit_delta = entry_price - curr_price
					realized_profit_delta_perc = realized_profit_delta / entry_price * 100
					cum_realized_profit += realized_profit_delta
					trade_row = list(
						np.concatenate([row.values,
						                [action, holding_period, realized_profit_delta, realized_profit_delta_perc,
						                 cum_realized_profit]]))
					trades = append_row(trades, trade_row)
					trade_position = TradePosition.NO_POSITION
			cum_continous_profit += daily_profit_delta
		
		continuos_profits.iloc[i, 1] = daily_profit_delta
		continuos_profits.iloc[i, 2] = cum_continous_profit
	
	return trades, continuos_profits


def append_row(trades, trade_row):
	trades = trades.append({DATE_COL_NAME: trade_row[0],
	                        PRICE_COL_NAME: trade_row[1],
	                        RESIDUAL_ZSCORE_COL_NAME: trade_row[2],
	                        ACTION_COL_NAME: trade_row[3].name,
	                        HOLDING_PERIOD_COL_NAME: trade_row[4],
	                        PROFIT_DELTA_COL_NAME: trade_row[5],
	                        PROFIT_DELTA_PERC_COL_NAME: trade_row[6],
	                        CUMULATIVE_PROFIT_COL_NAME: trade_row[7]}, ignore_index=True)
	return trades


def plot_trade_simuation(trades, continuos_profits):
	pass


def plot_regress_result(regress_data, regress_statistics):
	pass


def main():
	# calculate_cef_data(CEF_TICKER, START_DATE, AVERAGES_CALC_PERIOD)
	cef = read_processed_cef_data(CEF_TICKER)
	cef_train_data, cef_test_data = split_train_test_data(cef, TRAIN_DATA_RATIO)
	
	regress_data, regress_statistics = analyze_regression(cef_train_data, cef_test_data, NAV_RETURNS_COL_NAME, PRICE_RETURNS_COL_NAME)
	plot_regress_result(regress_data, regress_statistics)

# cef_simul_data = cef[[DATE_COL_NAME, PRICE_COL_NAME, PRICE_RETURNS_COL_NAME, NAV_RETURNS_COL_NAME]]
# simulation_start_date = cef_test_data[DATE_COL_NAME].values[0]
# cef_simul_data = calculate_residual_zscores(cef_simul_data, simulation_start_date, AVERAGES_CALC_PERIOD)

# trades, continuos_profits = run_trade_simulation(cef_simul_data)
# plot_trade_simuation(trades, continuos_profits)


if __name__ == "__main__":
	main()
