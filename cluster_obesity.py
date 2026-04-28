import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

#carregar dataset
dados = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', sep=',')

#NORMALIZAÇÃO

# Separar variáveis numéricas (contínuas) e categóricas (incluindo binárias)

# Variáveis numéricas (contínuas)
colunas_num = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
dados_num = dados[colunas_num]

# Todas as variáveis categóricas (binárias + multiclasse)
colunas_cat = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']
dados_cat = dados[colunas_cat]

scaler = MinMaxScaler() #instanciando normalizador

normalizador = scaler.fit(dados_num) #treinar normalizador com dados

pickle.dump(normalizador, open('normalizador_obesity.pkl', 'wb')) #salvar normalizador para uso posterior

dados_num_norm = normalizador.transform(dados_num) #normalizar os dados

# Normalizar os dados categóricos

dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int)

# converter a matriz numérica (dados_num_norm) em dataframe

dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

# juntar o dados_num_norm com dados_cat_norm

dados_dataframe = dados_num_norm.join(dados_cat_norm)

#HIPERPARAMETRIZAÇÃO - Determinar o número ótimos de clusters antes do treinamento

from sklearn.cluster import KMeans # kmeans é um clusterizador
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # método para cálculo de distâncias cartesianas
import numpy as np

distortions = [] #matriz para armazenar as distorções

K = range(1, 21)

for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_dataframe)

    # calcular e armazenar a distorção de cada treinamento
    distortions.append(
        sum(
            np.min(
                cdist(dados_dataframe, cluster_model.cluster_centers_, 'euclidean'),
                  axis=1)/dados.shape[0]
            )
        )
    
# Determinar o número ótimo de clusters para o modelo
x0 = K[0]
y0 = distortions[0]
xn = K[-1]
yn = distortions[-1]
distances = []
for i in range(len(distortions)):
    x = K[i]
    y= distortions[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distances.append(numerador/denominador)

numero_clusters_otimo = K[distances.index(np.max(distances))]

# Treinar o modelo com o número ótimo
cluster_model = KMeans(n_clusters= numero_clusters_otimo, random_state=42).fit(dados_dataframe)

#salvar o modelo para uso posterior
pickle.dump(cluster_model, open('cluster_obesity.pkl', 'wb'))

print(f"Número ótimo de clusters: {numero_clusters_otimo}")