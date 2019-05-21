import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import csv

benchmark_index_source = {"SPX": "spx_hist_prices.csv"}
cef_symbols_and_sources = {"ADX": ["adx_hist_prices.csv", "adx_hist_navs.csv"],
                           "CII": ["cii_hist_prices.csv", "cii_hist_navs.csv"],
                           "EOS": ["eos_hist_prices.csv", "eos_hist_navs.csv"]}

if __name__ == "__main__":
	
	with open(cef_symbols_and_sources["ADX"][0]) as csv_file:
		csv_reader = csv.reader(csv_file)
		for row in csv_reader:
			print(row)
	
	print()


# class HistQuote:
#
# 	def __init__(self, date):
# 		self._date = date
#
# 	def get_date(self):
# 		return self._date
#
#
# class HistPriceReturn(HistQuote):
#
# 	def __init__(self, date, price_return):
# 		HistQuote.__init__(self, date)
# 		self._price_return = price_return
#
# 	def get_price_return(self):
# 		return self._price_return
#
#
# class HistNavReturn(HistQuote):
#
# 	def __init__(self, date, nav_return):
# 		HistQuote.__init__(self, date)
# 		self._nav_return = nav_return
#
# 	def get_nav_return(self):
# 		return self._nav_return
#
#
# class HistPrice(HistQuote):
#
# 	def __init__(self, date, price):
# 		HistQuote.__init__(self, date)
# 		self._price = price
#
# 	def get_price(self):
# 		return self._price
#
#
# class HistNav(HistQuote):
#
# 	def __init__(self, date, nav):
# 		HistQuote.__init__(self, date)
# 		self._nav = nav
#
# 	def get_nav(self):
# 		return self._nav
