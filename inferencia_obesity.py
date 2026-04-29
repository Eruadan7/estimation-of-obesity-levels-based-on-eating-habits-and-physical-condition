import pandas as pd
import pickle

# Carregar as colunas do CSV salvo (38 colunas)
dados_normalizados = pd.read_csv('dados_normalizados_obesity.csv')
colunas_onehot = dados_normalizados.columns.tolist()
colunas_num = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# novo dado com 38 valores (incluindo target como zeros)
novo_dado = [
    # 8 numéricas
    22, 1.65, 60, 3, 3, 2, 1, 1,
    
    # Gender (2)
    1, 0,  # Female=1, Male=0
    
    # family_history_with_overweight (2)
    0, 1,  # no=0, yes=1
    
    # FAVC (2)
    1, 0,  # no=1, yes=0
    
    # SMOKE (2)
    1, 0,  # no=1, yes=0
    
    # SCC (2)
    1, 0,  # no=1, yes=0
    
    # CAEC (4)
    0, 0, 1, 0,  # Always=0, Frequently=0, Sometimes=1, no=0
    
    # CALC (4)
    0, 0, 0, 1,  # Always=0, Frequently=0, Sometimes=0, no=1
    
    # MTRANS (5)
    0, 0, 0, 1, 0,  # Automobile=0, Bike=0, Motorbike=0, Public=1, Walking=0
    
    # TARGET (7 colunas NObeyesdad_*)
    0, 0, 0, 0, 0, 0, 0
]

print(f"Total de valores fornecidos: {len(novo_dado)}")  # Deve ser 38

# abrir normalizador e modelo
normalizador = pickle.load(open('normalizador_obesity.pkl', 'rb'))
cluster_obesity = pickle.load(open('cluster_obesity.pkl', 'rb'))

# Criar DataFrame
novo_dataframe = pd.DataFrame([novo_dado], columns=colunas_onehot)

# Normalizar APENAS as colunas numéricas
novo_dataframe[colunas_num] = normalizador.transform(novo_dataframe[colunas_num])

# Classificar
cluster = cluster_obesity.predict(novo_dataframe)
print(f"Cluster do novo paciente: {cluster[0]}")