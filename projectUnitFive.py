import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# Download historical data for a stock
msft = yf.Ticker("MSFT")
msft_data = msft.history(period="max")
# Display the downloaded data
#print(msft_data.head())
#msft_data.reset_index(inplace=True)
#msft_data.plot(x="Date", y="Open")
#plt.show()
print(msft.dividends)
msft.dividends.plot()
plt.show()