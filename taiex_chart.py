"""Plot the previous 30 days of the TAIEX index level and volume."""

import yfinance as yf
import mplfinance as mpf

# ^TWII is the Taiwan Weighted Index (TAIEX) on Yahoo Finance.
data = yf.download("^TWII", period="3mo", auto_adjust=False)

# yfinance can return MultiIndex columns; flatten to single level.
if isinstance(data.columns, __import__("pandas").MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Keep the most recent 30 trading days.
data = data.tail(30)

mpf.plot(
    data,
    type="candle",
    volume=True,
    style="yahoo",
    title="TAIEX (^TWII) - Last 30 Trading Days",
    ylabel="Index Level",
    ylabel_lower="Volume",
)
