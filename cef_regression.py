import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns

TRAIN_DATA_COEFFICIENT = 0.75
AVERAGES_CALC_PERIOD = relativedelta(months=1)
START_DATE = datetime.datetime(2018, 5, 13)

PREM_DISC_ZSCORE_COL_NAME = "prem/disc z-score"
PRICE_RETURNS_COL_NAME = "return on price"
NAV_RETURNS_COL_NAME = "return on nav"

BENCHMARK_INDEX_SOURCE = {"SPX": "spx_hist_prices.csv"}
CEF_DATA_SOURCES = {"ADX": ["adx_hist_prices.csv", "adx_hist_navs.csv"],
                    "CII": ["cii_hist_prices.csv", "cii_hist_navs.csv"],
                    "EOS": ["eos_hist_prices.csv", "eos_hist_navs.csv"]}


def read_cef_data(cef_symbol):
	cef = pd.read_csv(CEF_DATA_SOURCES[cef_symbol][0])
	cef = cef[["timestamp", "close"]]
	cef.columns = ["date", "price"]
	cef["nav"] = pd.read_csv(CEF_DATA_SOURCES[cef_symbol][1])["close"]
	cef["date"] = pd.to_datetime(cef["date"])
	cef = cef.sort_values(["date"])
	return cef


def read_index_data(index_symbol):
	index = pd.read_csv(BENCHMARK_INDEX_SOURCE[index_symbol])
	index = index[["timestamp", "close"]]
	index.columns = ["date", "price"]
	index["date"] = pd.to_datetime(index["date"])
	index = index.sort_values(["date"])
	return index


def calculate_zscore(cef, period_start_date, period_end_date):
	prem_discs = cef.loc[(cef["date"] >= period_start_date) & (cef["date"] <= period_end_date), "prem/disc"]
	curr_prem_disc = cef.loc[cef["date"] == period_end_date, "prem/disc"].values[0]
	average_prem_disc = prem_discs.mean()
	std_prem_disc = prem_discs.std()
	# cef.loc[cef["date"] == period_end_date, "nav mean"] = average_prem_disc
	# cef.loc[cef["date"] == period_end_date, "nav std"] = std_prem_disc
	cef.loc[cef["date"] == period_end_date, PREM_DISC_ZSCORE_COL_NAME] = (
			                                                                     curr_prem_disc - average_prem_disc) / std_prem_disc
	return cef


def calculate_price_return(cef, period_start_date, period_end_date):
	base_price = cef.loc[cef["date"] == period_start_date, "price"].values[0]
	curr_price = cef.loc[cef["date"] == period_end_date, "price"].values[0]
	price_return = (curr_price - base_price) / base_price * 100
	cef.loc[cef["date"] == period_end_date, PRICE_RETURNS_COL_NAME] = price_return
	return cef


def calculate_nav_return(cef, period_start_date, period_end_date):
	base_nav = cef.loc[cef["date"] == period_start_date, "nav"].values[0]
	curr_nav = cef.loc[cef["date"] == period_end_date, "nav"].values[0]
	nav_return = (curr_nav - base_nav) / base_nav * 100
	cef.loc[cef["date"] == period_end_date, NAV_RETURNS_COL_NAME] = nav_return
	return cef


def find_valid_period_start_date(dates, date, period):
	period_start_date = date - period
	period_dates = dates[dates >= period_start_date]
	first_date = period_dates.iloc[0]
	return first_date


def calculate_factors(cef, start_date, period):
	cef = cef[(cef["date"] >= start_date - period)]
	all_dates = cef["date"]
	cef["prem/disc"] = (cef["price"] - cef["nav"]) / cef["nav"] * 100
	
	dates = cef.loc[cef["date"] >= start_date, "date"]
	for date in dates:
		period_start_date = find_valid_period_start_date(all_dates, date, period)
		cef = calculate_price_return(cef, period_start_date, date)
		cef = calculate_nav_return(cef, period_start_date, date)
		cef = calculate_zscore(cef, period_start_date, date)
		print(date)
	return cef


def calculate_cef_data(symbol, start_date, calc_period):
	symbol_df = read_cef_data(symbol)
	symbol_df = calculate_factors(symbol_df, start_date, calc_period)
	symbol_df = symbol_df.loc[symbol_df["date"] >= start_date]
	symbol_df = symbol_df[
		["date", PRICE_RETURNS_COL_NAME, NAV_RETURNS_COL_NAME, PREM_DISC_ZSCORE_COL_NAME]].reset_index(drop=True)
	file_name = symbol.lower() + "_data.csv"
	symbol_df.to_csv(file_name)


def split_train_test_data(cef, split_data_coefficient):
	train_data_end_index = int((len(cef) - 1) * split_data_coefficient)
	cef_train_data = cef.loc[cef.index <= train_data_end_index]
	cef_test_data = cef.loc[cef.index > train_data_end_index].reset_index(drop=True)
	return cef_train_data, cef_test_data


def analyze_regression(cef_train_data, cef_test_data, regressor_col_name, regressand_col_name):
	regressor_train_data = cef_train_data[regressor_col_name].values.reshape(-1, 1)
	regressand_train_data = np.array(cef_train_data[regressand_col_name].values)
	
	regress_model = LinearRegression().fit(regressor_train_data, regressand_train_data)
	
	regressor_test_data = cef_test_data[regressor_col_name].values.reshape(-1, 1)
	regressand_test_data = np.array(cef_test_data[regressand_col_name].values)
	
	corr_train_data = cef_train_data[regressor_col_name].corr(cef_train_data[regressand_col_name])
	
	regressand_predicted_test_data = regress_model.predict(regressor_test_data)
	
	corr_predicted_actual = np.corrcoef(regressand_predicted_test_data, regressand_test_data)[0, 1]
	predicted_actual_deltas = regressand_test_data - regressand_predicted_test_data
	
	plt.scatter(regressand_predicted_test_data, regressand_test_data)
	plt.show()
	dates = cef_test_data["date"]
	plt.bar(dates, predicted_actual_deltas)
	plt.xticks(rotation=90)
	plt.show()
	plt.plot(dates, regressand_predicted_test_data, label="Predicted")
	plt.plot(dates, regressand_test_data, label="Actual")
	plt.legend()
	plt.xticks(rotation=90)
	plt.show()
	
	print("y = {0:.3f} + {1:.3f}*x".format(regress_model.intercept_, regress_model.coef_[0]))
	print("corr_train_data = {0:.3f}".format(corr_train_data))
	print("R-square = {0:.3f}".format(regress_model.score(regressor_train_data, regressand_train_data)))
	print("corr_predicted_actual = {0:.3f}".format(corr_predicted_actual))


def main():
	# calculate_cef_data("ADX", START_DATE, AVERAGES_CALC_PERIOD)
	
	adx = pd.read_csv("adx_data.csv", index_col=0)
	cef_train_data, cef_test_data = split_train_test_data(adx, TRAIN_DATA_COEFFICIENT)
	# print(cef_train_data.corr())
	
	# analyze_regression(cef_train_data, cef_test_data, PREM_DISC_ZSCORE_COL_NAME, PRICE_RETURNS_COL_NAME)
	analyze_regression(cef_train_data, cef_test_data, NAV_RETURNS_COL_NAME, PRICE_RETURNS_COL_NAME)


if __name__ == "__main__":
	main()
