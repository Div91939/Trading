import pandas as pd
import yfinance as yf
import numpy as np

TICKER         = "dolatalgo.bo"
stock = yf.Ticker(TICKER)
info  = stock.info
csv_path='C:\\Users\\Divyansh\\OneDrive\\Desktop\\IISER\\Trading\\Data\\dolat.csv'


company_name  = info.get('longName') or info.get('shortName') or TICKER
current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

df=pd.DataFrame((stock.history(period="1y", interval="1d")))
avg_volume=np.array(info.get('averageDailyVolume10Day'))

date=np.array(df.index)

df.insert(0, 'Date', pd.to_datetime(df.index).tz_localize(None).strftime('%d-%m-%Y'))

df.insert(5,'Avg_Volume', avg_volume)
df.to_csv(csv_path, index=False)
