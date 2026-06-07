# Technical Price Indicator Engineering

Using the original daily prices and volumes from Yahoo! Finance, many other technical price
indicators are derived. The full list of indicators is given here:

- Feature (Symbol) Formula
- High (H) Raw data from Yahoo! Finance
- Close (C) Raw data from Yahoo! Finance
- Open (O) Raw data from Yahoo! Finance
- Low (L) Raw data from Yahoo! Finance
- Adj Close (AC) Raw data from Yahoo! Finance
- Volume (V) Raw data from Yahoo! Finance
- Amplitude (Ht – Lt) / ACt-1
- Difference (Ct – Ot) / ACt-1
- Intraday (I) Ot-1 – ACt-1
- ∆Adj Close (∆AC) ACt – ACt-1
- ∆Volume (∆V) Vt – Vt-1
- Intraday MAn n-day Moving Average of I
- ∆Adj Close MAn n-day Moving Average of ∆AC
- ∆Volume MAn n-day Moving Average of ∆V
- ±∆Open 1 if Ot – Ot-1 >0, else -1
- Daily Return (DR) %∆C
- Cumulative Daily
- Return
- Cumulative product of DR
- H-L Ht – Lt
- C-O Ct – Ot
- RSI 14-day period of Relative Strength Index on C
- Williams %R (Hmax – Ct) / (Hmax – Lmin) \* –100
- MAn n-day Moving Average of C
- EMAn n-day Exponential Moving Average of C
- MACD EMA12 – EMA26
- BB High 21-day Cavg + 2 \* 21-day Cstd
- BB Low 21-day Cavg – 2 \* 21-day Cstd
- EMA Exponential moving average of C with 0.5 decay
- Momentum Ct – 1
- Feature (Symbol) Formula
- High (H) Raw data from Yahoo! Finance
- Close (C) Raw data from Yahoo! Finance
- Open (O) Raw data from Yahoo! Finance
- Low (L) Raw data from Yahoo! Finance
- Adj Close (AC) Raw data from Yahoo! Finance
- Volume (V) Raw data from Yahoo! Finance
