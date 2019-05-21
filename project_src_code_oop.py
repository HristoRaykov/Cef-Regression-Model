import numpy as np
import datetime
import csv

start_date = datetime.datetime(2009, 5, 13)
benchmark_index_source = {"SPX": "spx_hist_prices.csv"}
cef_symbols_and_sources = {"ADX": ["adx_hist_prices.csv", "adx_hist_navs.csv"],
                           "CII": ["cii_hist_prices.csv", "cii_hist_navs.csv"],
                           "EOS": ["eos_hist_prices.csv", "eos_hist_navs.csv"]}


def parse_hist_prices(file_name, start_date):
	hist_prices = {}
	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file)
		next(csv_reader, None)
		for row in csv_reader:
			date = datetime.datetime.strptime(row[0], "%Y-%m-%d")
			close_price = float(row[4])
			hist_prices[date] = close_price
	
	return {k: hist_prices[k] for k in sorted(hist_prices.keys()) if k >= start_date}


def parse_hist_navs(file_name, start_date):
	hist_navs = {}
	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file)
		next(csv_reader, None)
		for row in csv_reader:
			date = datetime.datetime.strptime(row[0], "%Y-%m-%d")
			nav = float(row[4])
			hist_navs[date] = nav
	
	return {k: hist_navs[k] for k in sorted(hist_navs.keys()) if k >= start_date}


def parse_index_data(symbol, file_name, start_date):
	hist_prices = parse_hist_prices(file_name, start_date)
	spx = Stock(symbol, hist_prices)
	return spx


def parse_cef_data(symbol, prices_source_file, navs_source_file, start_date):
	hist_prices = parse_hist_prices(prices_source_file, start_date)
	hist_navs = parse_hist_navs(navs_source_file, start_date)
	cef = Cef(symbol, hist_prices, hist_navs)
	return cef


class Stock:
	
	def __init__(self, symbol, hist_prices):
		self._symbol = symbol
		self._hist_prices = hist_prices
		self._daily_price_returns = {}
		self.calculate_daily_price_returns()
	
	def get_symbol(self):
		return self._symbol
	
	def get_hist_prices(self):
		return self._hist_prices
	
	def get_daily_price_returns(self):
		return self._daily_price_returns
	
	def calculate_daily_price_returns(self):
		dates = list(self._hist_prices.keys())
		for i in range(0, len(dates) - 1):
			date_t0 = dates[i]
			date_t1 = dates[i + 1]
			daily_price_return = (self._hist_prices[date_t1] - self._hist_prices[date_t0]) * 100 / self._hist_prices[
				date_t0]
			self._daily_price_returns[date_t1] = daily_price_return


class Cef(Stock):
	
	def __init__(self, symbol, hist_prices, hist_navs):
		Stock.__init__(self, symbol, hist_prices)
		self._hist_navs = hist_navs
		self._daily_nav_returns = {}
		self._prem_discs = {}
		self.calculate_factors()
	
	def get_hist_navs(self):
		return self._hist_navs
	
	def get_daily_nav_returns(self):
		return self._daily_nav_returns
	
	def get_prem_discs(self):
		return self._prem_discs
	
	def calculate_daily_nav_returns(self):
		dates = list(self._hist_navs.keys())
		for i in range(0, len(dates) - 1):
			date_t0 = dates[i]
			date_t1 = dates[i + 1]
			daily_nav_return = (self._hist_navs[date_t1] - self._hist_navs[date_t0]) * 100 / self._hist_navs[date_t0]
			self._daily_nav_returns[date_t1] = daily_nav_return
	
	def calculate_factors(self):
		self.calculate_daily_nav_returns()
		self.calculate_prem_discs()
	
	def calculate_prem_discs(self):
		dates = list(self.get_hist_prices().keys())
		for date in dates:
			price = 0.0
			nav = 0.0
			if date in self._hist_prices:
				price = self._hist_prices[date]
			else:
				continue
			if date in self._hist_navs:
				nav = self._hist_navs[date]
			else:
				continue
			prem_disc = (price - nav) * 100 / nav
			self._prem_discs[date] = prem_disc


def main():
	spx = parse_index_data("SPX", benchmark_index_source["SPX"], start_date)
	adx = parse_cef_data("ADX", cef_symbols_and_sources["ADX"][0], cef_symbols_and_sources["ADX"][1], start_date)
	
	print()


if __name__ == "__main__":
	main()
