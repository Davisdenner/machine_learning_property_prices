# -*- coding: utf-8 -*-
"""Preços de Imóveis Utilizando Regressão Linear.ipynb

# Preço de imoveis

**Objetivo**: estimar os preços de imóveis
 * Identificar aspectos que contribuem para precificação dos imoveis
 * Entender qual aspecto é mais relevante, qual influencia mais no preço do imóvel.
 * Precificar um imóvel novo.

Base de dados simplificada e inspirada em [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)







## Conhecendo os dados
"""

import pandas as pd

#Lendo dados
dados = pd.read_csv("/content/Preços_de_casas.csv")

#Quais fatores coletados?

dados.info()

dados = dados.drop(columns = "Id")

"""## Correlação

Quais fatores estão relacionados ao preço da casa? Como é essa relação?

Com o coeficiente de Correlação de Pearson nos permite medir a relação linear entre variáveis, oferecendo uma escala que varia de -1 a 1, que interpretamos conforme sua intensidade e direção:

* -1: correlação positiva perfeita: à medida que uma variável aumenta, a outra também aumenta.
* 0: não há relação linear entre as variáveis.
* 1: indica uma correlação negativa perfeita: à medida que uma variável aumenta, a outra também diminui.


"""

# Correlação
corr = dados.corr()

corr['preco_de_venda']

# Atividade
# Quais fatores estão mais correlacionados?

"""## Relacionando variáveis"""

# importando as visualizações
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Como é a relação entre área construida e o preço do imóvel?
plt.scatter(dados['area_primeiro_andar'], dados['preco_de_venda'])
plt.title("Relação entre Preço e Area")
plt.xlabel("Area do primeiro andar")
plt.ylabel("Preço de venda")

# Aparentemente quanto maior a área do primeiro andar, maior o preço da casa.
# E se quisermos traçar uma linha que melhor representa esse comportamento?
plt.scatter(dados['area_primeiro_andar'], dados['preco_de_venda'])
plt.axline(xy1 = (66, 250000),xy2 = (190, 1800000), color = "red" )
plt.title("Relação entre Preço e Area")
plt.xlabel("Area do primeiro andar")
plt.ylabel("Preço de venda")

"""## Melhor reta"""

# Qual a reta que melhor se adequa a relação?
px.scatter(dados, x = 'area_primeiro_andar', y = 'preco_de_venda', trendline_color_override="red", trendline = 'ols' )

"""#Explicando a reta
Ajustamos uma reta entre o $m^2$ do primeiro andar e o preço da casa. Queremos explicar o preço da casa a partir do seu tamanho, por isso dizemos que:

* Variável explicativa/independente: Área do primeiro andar

* Variável resposta/dependente: Preço da casa
"""

#Quem é nossa variável resposta?

sns.displot(dados['preco_de_venda'], kde=True, color='green')
plt.title('Distribuição do preço de venda')
plt.show()

"""### Separando em treino e teste

O conjunto de **treinamento** é usado para ajustar o modelo, enquanto o conjunto de **teste** é usado para avaliar seu desempenho em prever preços de casas não vistos durante o treinamento, que auxilia na generalização do modelo.
"""

# import train_test_split
from sklearn.model_selection import train_test_split

# Definindo y e X
y = dados['preco_de_venda']
X = dados.drop(columns = 'preco_de_venda')

#Aplicando o split do y e X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 230)

#Dados de treino para usar a fórmula

df_train = pd.DataFrame(data= X_train )
df_train['preco_de_venda'] = y_train

# import ols
from statsmodels.formula.api import ols

# ajustando o primeiro modelo
modelo_0 = ols('preco_de_venda ~ area_primeiro_andar', data = df_train ).fit()

"""## Analisando os coeficientes

(intensidade, direção e significância)



"""

# visualizando os parametros

modelo_0.params

# o resumo do nosso modelo
print(modelo_0.summary())

"""* O **intercepto** é o valor esperado do preço da casa quando todas as outras variáveis são zero. Neste caso, quando todas as outras variáveis a área do primeiro andar é zero, o preço esperado da casa é de R$ 145.196,40. Nem sempre temos uma interpretação prática desse número.

* O **efeito individual** da área é 6833.97. Isso indica que para cada 1m² adicinado à área do primero andar espera-se que o preço da casa aumente em média R$6.833,97.

<img src="https://i.imgur.com/7Cm4Cot.png" width="500"/>

## Explicabilidade do modelo

Quanto a variação da área está explicando os diversos preços das casas?

Nesse caso recorremos a métrica R², o coeficiente de determinação. O R² varia de 0 a 1, onde 1 indica um ajuste perfeito do modelo aos dados, ou seja, todas as variações na variável dependente são explicadas pelas variáveis independentes no modelo. Por outro lado, um R² de 0 indica que o modelo não explica nenhuma variabilidade na variável dependente
"""

# observando o R²
modelo_0.rsquared

"""## Entendendo o resíduo"""

# Quem são os residuos
modelo_0.resid

# Como eles estão distribuidos
modelo_0.resid.hist()
plt.title("Distribuição dos residuos")
plt.show()





"""![](https://i.imgur.com/CJMdXpf.png)

## Obtendo o R² da previsão
"""

# definindo o Y previsto
y_predict = modelo_0.predict(X_test)

# importando o r2_score
from sklearn.metrics import r2_score

# printando o r²
print("R²: ", r2_score(y_test,y_predict ))

"""#Adicionando outras características

O modelo com apenas um fator nos mostrou um R² de 0.37, ou seja, aproximadamente 37% da variação observada nos preços das casas pode ser explicada pela variação na área.
Isso indica que ainda há uma quantidade significativa de variação que não está sendo capturada por esse modelo específico.Vamos analisar outros fatores para explicar o preço das casas.

## Analisando os fatores
"""

# quais outras características poderiam explicar o preço dos imóveis?
sns.pairplot(dados)

dados.columns

#Vamos olhar apenas com y_vars='preco_de_venda'
sns.pairplot(dados, y_vars = 'preco_de_venda', x_vars = ['quantidade_banheiros','area_segundo_andar','capacidade_carros_garagem'] )

"""## Adicionando fatores no modelo"""

# importando a api do statsmodels
import statsmodels.api as sm

# adicionando o constante
X_train = sm.add_constant(X_train)

X_train.head()

X_train.columns

# Criando o modelo de regressão (sem fómula): saturado
modelo_1 = sm.OLS(y_train,
                  X_train[['const','area_primeiro_andar','existe_segundo_andar',
                          'area_segundo_andar','quantidade_banheiros','capacidade_carros_garagem',
                           'qualidade_da_cozinha_Excelente']]).fit()

# Modelo sem a área do segundo andar
modelo_2 = sm.OLS(y_train,
                  X_train[['const','area_primeiro_andar','existe_segundo_andar',
                          'quantidade_banheiros','capacidade_carros_garagem',
                           'qualidade_da_cozinha_Excelente']]).fit()

# Modelo sem informações sobre garagem
modelo_3 = sm.OLS(y_train,
                  X_train[['const','area_primeiro_andar','existe_segundo_andar',
                          'quantidade_banheiros',
                           'qualidade_da_cozinha_Excelente']]).fit()

# Resumo do modelo 1
print(modelo_1.summary())

# Resumo do modelo 2
print(modelo_2.summary())

# Resumo do modelo 3
print(modelo_3.summary())

"""## Comparando modelos
Qual o melhor modelo?

"""

print("R²")
print("Modelo 0: ", modelo_0.rsquared)
print("Modelo 1: ", modelo_1.rsquared)
print("Modelo 2: ", modelo_2.rsquared)
print("Modelo 3: ", modelo_3.rsquared)

#Quantos parametros estão no modelo?
print(len(modelo_0.params))
print(len(modelo_1.params))
print(len(modelo_2.params))
print(len(modelo_3.params))

modelo_3.params

"""#Precificando as casas

## Obtendo o R² da previsão
"""

X_test.columns

modelo_3.params

# Adicionando uma constante em X_test
X_test = sm.add_constant(X_test)

# Prevendo com o modelo 3
predict_3 = modelo_3.predict(X_test[['const','area_primeiro_andar','existe_segundo_andar', 'quantidade_banheiros',
                         'qualidade_da_cozinha_Excelente' ]])

# Qual o r² da previsão?
modelo_3.rsquared

# Qual o R² do treino?
print("R²: ", r2_score(y_test, predict_3))

"""## Precificando uma casa


<img src="https://i.imgur.com/e4gytI1.png" width="800"/>
"""

modelo_3.params

#Novo imovel
novo_imovel = pd.DataFrame({ 'const': [1],
                             'area_primeiro_andar': [120],
                              'existe_segundo_andar': [1],
                              'quantidade_banheiros': [2],
                              'qualidade_da_cozinha_Excelente':[0]
})

# Qual o preço desse imóvel com o modelo 0?
modelo_0.predict(novo_imovel['area_primeiro_andar'])

# Qual o preço desse imóvel com o modelo 3?
print(modelo_3.predict(novo_imovel)[0])

"""## Precificando várias casas


"""

# Lendo várias casas?
novas_casas = pd.read_csv("/content/Novas_casas.csv", sep = ";")

novas_casas.head()

novas_casas = novas_casas.drop(columns = "Casa" )

# Adicionando uma constante
novas_casas = sm.add_constant(novas_casas)

# Qual o preço dessas novas casas?
modelo_3.predict(novas_casas)