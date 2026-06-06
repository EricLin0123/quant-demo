import yfinance as yf
import mplfinance as mpf

# 抓取臺灣加權指數歷史數據（多抓一些以確保有足夠的交易日）
taiwan_index = yf.Ticker("^TWII")
df = taiwan_index.history(period="3mo")

# 取最近 30 個交易日
df = df.tail(30)

# 繪製日 K 線圖（一天一根蠟燭）
mpf.plot(
    df,
    type="candle",
    style="yahoo",
    title="Taiwan Index - Last 30 Trading Days (Daily)",
    ylabel="Price",
    volume=True,
)
