from matplotlib import pyplot as plt
import pandas as pd
from fbprophet import Prophet

# LÊ O ARQUIVO DO RESERVATÓRIO
df = pd.read_csv("dados/sobradinho.csv",  header=2, delimiter='\t',
                 encoding='utf-16', usecols=[0,8], names=['ds','y'],
                 parse_dates=['ds'], dayfirst=True, decimal=',')

print(df.head())

# Atribui a série temporal a ser avaliada na biblioteca Prophet()
model = Prophet()
model.fit(df)

# Cria uma série de tempo de 365 dias, que será a base de predição
future = model.make_future_dataframe(periods=365)
future.tail()
print(future.tail())

# Faz a predição baseada na série temporal atribuida.
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig1 = model.plot(forecast, ylabel='Profundidade', xlabel='Data')

ax = fig1.gca()
ax.set_title("Previsão para o Reservatório de Sobradinho", size=14)

plt.show()




