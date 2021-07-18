from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd


# LÊ O ARQUIVO DO RESERVATÓRIO DE SOBRADINHO
df = pd.read_csv("dados/sobradinho.csv",  header=2, delimiter='\t', index_col=0,
                 encoding='utf-16', usecols=[0,8], names=['ds','y'],
                 parse_dates=['ds'], dayfirst=True, decimal=',')

df.index = df.index.to_period('D')

#df = df["2004-01-01":"2005-01-01"]
print(df.head())

"""
from statsmodels.tsa.stattools import adfuller
print("p-value:", adfuller(df)[1])

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df.diff().diff().dropna(), lags=40, ax=ax1,title="ACF")
ax2 = fig.add_subplot(212)
fig = plot_pacf(df.diff().diff().dropna(), lags=40, ax=ax2, title="PACF")

plt.show()
"""

arima = ARIMA(df, order=(8,0,1), seasonal_order=(0,0,1,12))

model_fit = arima.fit()

print(model_fit.summary())

prediction = model_fit.get_prediction(start='2021-03-01',end='2022-03-01')

predicted_values = prediction.predicted_mean
confidence_intervals = prediction.conf_int(alpha=0.05)

df.plot()
predicted_values.plot()
plt.xlabel("Data")
plt.ylabel("Profundidade")
plt.title("Previsão da Profundidade do Reservatório de Sobradinho")
plt.legend(("passado", "previsão"), loc="upper left")
plt.fill_between(confidence_intervals.index.values,
                 confidence_intervals["lower y"][0:].values,
                 confidence_intervals["upper y"][0:].values, color="red", alpha=0.25)
plt.show()