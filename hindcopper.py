import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import yfinance as yf
from email.mime.text import MIMEText

ticker       = "hindcopper.bo"
csv_path      = r'C:\\Users\\Divyansh\\OneDrive\\Desktop\\IISER\\Trading\\Data\\hindcopper.csv'
#LOG_PATH       = r'C:\\Users\\Divyansh\\OneDrive\\Desktop\\IISER\\Finance_Codes\\email_log.json'
email   = "divyanshdewan@gmail.com"
key = "osrp rtab jvyv rcvz"


df=pd.read_csv(csv_path)

stock = yf.Ticker(ticker)
info  = stock.info

company_name  = info.get('longName') or info.get('shortName')
current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
avg_volume    = info.get('averageDailyVolume10Day') or 0
volume        = info.get('volume') or 0

hist_today = stock.history(period="1d", interval="1d")
today_str = hist_today.index[-1].strftime('%d-%m-%Y')

new_row = {
    'Date':         today_str,
    'Open':         round(float(hist_today['Open'].iloc[-1]),  2),
    'High':         round(float(hist_today['High'].iloc[-1]),  2),
    'Low':          round(float(hist_today['Low'].iloc[-1]),   2),
    'Close':        round(float(hist_today['Close'].iloc[-1]), 2),
    'Volume':       volume,
    'Avg_Volume':   avg_volume,
    'Dividends':    round(float(hist_today['Dividends'].iloc[-1]), 2),
    'Stock Splits': round(float(hist_today['Stock Splits'].iloc[-1]), 2),
}

df = df.dropna(how='all').drop_duplicates(subset='Date', keep='last')
df.columns = df.columns.str.strip()

if today_str in df['Date'].values:
    idx = df.index[df['Date'] == today_str][0]
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg_Volume']:
        df.loc[idx, col] = new_row[col]
else:
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(csv_path, index=False)
df = pd.read_csv(csv_path)
open=df['Open']
close=df['Close']
high=df['High']
low=df['Low']
volume=df['Volume']
x_range = [i for i in range(len(close),0,-1)]
signal_rsi=np.zeros(len(close))
signal_bb=np.zeros(len(close))

rsi = np.array(ta.momentum.RSIIndicator(df["Close"], window=14).rsi())
bb      = ta.volatility.BollingerBands(df["Close"], window=25, window_dev=2)
bb_up   = np.array(bb.bollinger_hband())
bb_mav  = np.array(bb.bollinger_mavg())
bb_low  = np.array(bb.bollinger_lband())
    
def signals():
    for i in range(len(close)):
        if rsi[i]<35:
            signal_rsi[i]+=close[i]
        elif close[i]<= 1.05*bb_low[i]:
            signal_bb[i]+=close[i]
            
signals()
print(signal_rsi)

x_range = [i for i in range(len(close),0,-1)]
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(14, 10),gridspec_kw={'height_ratios': [4, 1]},sharex=True)
ax1.plot(x_range,close, color='blue')
ax1.plot(x_range,bb_mav, color='black')
ax1.plot(x_range,bb_up, color='green')
ax1.plot(x_range,bb_low, color='red')
ax1.scatter(x_range,signal_rsi,color='pink',s=100)
ax1.set_xlim(len(close)/2, 1)
ax1.set_ylim(300,700)
ax1.grid(True)

ax2.plot(x_range, rsi)
ax2.set_xlim(len(close)/2, 1)
ax2.grid(True)
ax2.axhspan(10, 35, color='red', alpha=0.2)
ax2.axhspan(80,100, color='green', alpha=0.2)

plt.tight_layout()
plt.show()