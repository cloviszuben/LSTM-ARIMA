"""
PROGRAMA GRÁFICO - Gera os gráficos que com os dados originais da coleta a fim de dar subsídio a análise.
"""

import matplotlib.pyplot as plt
import pandas as pd

# LÊ O ARQUIVO DO RESERVATÓRIO - Substitua o nome do arquivo .CSV para o nome do reservatório desejado.

df = pd.read_csv("dados/furnas.csv", index_col=0, header=2, delimiter='\t', encoding='utf-16', usecols=[0,6,8], names=['data','nome','profundidade'],
                 parse_dates=['data'], dayfirst=True, decimal=',')
df.info()
print (df)

# IMPRIME O GRÁFICO DA SÉRIE TEMPORAL

df.plot(color='r', title='Reservatório Furnas', y=['profundidade'], xlabel='Data', linewidth=1, kind='line')
plt.grid()
plt.show()


# LÊ O ARQUIVO DO RESERVATÓRIO - Substitua o nome do arquivo .CSV para o nome do reservatório desejado.

df = pd.read_csv("dados/belomonte.csv", index_col=0, header=2, delimiter='\t', encoding='utf-16', usecols=[0,6,8], names=['data','nome','profundidade'],
                 parse_dates=['data'], dayfirst=True, decimal=',')
df.info()
print (df)

# IMPRIME O GRÁFICO DA SÉRIE TEMPORAL

df.plot(color='r', title='Reservatório Belo Monte', y=['profundidade'], xlabel='Data', linewidth=1, kind='line')
plt.grid()
plt.show()



# LÊ O ARQUIVO DO RESERVATÓRIO - Substitua o nome do arquivo .CSV para o nome do reservatório desejado.

df = pd.read_csv("dados/sobradinho.csv", index_col=0, header=2, delimiter='\t', encoding='utf-16', usecols=[0,6,8], names=['data','nome','profundidade'],
                 parse_dates=['data'], dayfirst=True, decimal=',')
df.info()
print (df)

# IMPRIME O GRÁFICO DA SÉRIE TEMPORAL

df.plot(color='r', title='Reservatório Sobradinho', y=['profundidade'], xlabel='Data', linewidth=1, kind='line')
plt.grid()
plt.show()
