# imports
import pickle
import pandas as pd
import numpy as np

# abrir modelo de clusters
cluster_model = pickle.load(open('cluster_obesity.pkl', 'rb'))

# abrir normalizador numérico salvo anteriormente
normalizador = pickle.load(open('normalizador_obesity.pkl', 'rb'))

# DESNORMALIZAR OS CENTRÓIDES

# obter os nomes das colunas
dados_dataframe = pd.read_csv('dados_normalizados_obesity.csv')
columns_names = dados_dataframe.columns.tolist()

# identificar colunas numéricas (as mesmas usadas no treinamento do scaler)
colunas_num = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
colunas_cat = [col for col in columns_names if col not in colunas_num]

# converter os centroides em dataframe
centroides_df = pd.DataFrame(cluster_model.cluster_centers_,
                             columns=columns_names)

# Desnormalizar APENAS as colunas numéricas
centroides_num = normalizador.inverse_transform(centroides_df[colunas_num])
centroides_num_df = pd.DataFrame(centroides_num, 
                                  columns=colunas_num,
                                  index=centroides_df.index)

# Juntar com as colunas categóricas (que não precisam desnormalizar)
centroides_desnorm = centroides_num_df.join(centroides_df[colunas_cat])

print("=== Centróides desnormalizados (apenas numéricas) ===")
print(centroides_desnorm)

#salvando
centroides_desnorm.to_csv('centroides_desnormalizados.csv', index=False)