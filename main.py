import pandas as pd
import numpy as np
import pickle
import math
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

CSV_PATH = 'dataset/ObesityDataSet_raw_and_data_sinthetic.csv'
PKL_PATH = 'obesity_cluster.pkl'
NOR_PATH = 'obsity_normalizer.pkl'
K_RANGE = range(1,501) # De 1 a 100 clusters (voltar depois)

# Ler o csv

df = pd.read_csv(CSV_PATH, sep=',', decimal='.') ####

# SEparação das colunas numéricas | categóricas

colunas_num = df.select_dtypes(exclude=['object']).columns.tolist()
colunas_cat = df.select_dtypes(include=['object']).columns.tolist()

dados_num = df[colunas_num]
dados_cat = df[colunas_cat]

# Normalização

scaler = MinMaxScaler()
normalizer = scaler.fit(dados_num)
pickle.dump(normalizer, open(NOR_PATH, 'wb'))

dados_num_norm = normalizer.transform(dados_num)
dados_num_norm = pd.DataFrame(dados_num_norm, columns=colunas_num)

#TESTE (APAGAR)
print('Amostra normalizada (primeiras 3 linhas): ')
print(dados_num_norm.head(3).to_string())
print()

# Codificas Dados Cat

dados_cat_norm = pd.get_dummies(dados_cat, dtype=int)


# Join nos dados normalizados

dados_norm = dados_num_norm.join(dados_cat_norm, how='left')

# Elbow method 

def calcular_distorcao(cluster, data):
    distancias = cdist(data, cluster.cluster_centers_, 'euclidean')
    menor_dist = np.min(distancias, axis=1)
    return sum(menor_dist) / data.shape[0]

def calcular_comprimento_reta(k_range, distorcoes):
    x0, y0 = k_range[0], distorcoes[0]
    xn, yn = k_range[-1], distorcoes[-1]
    return math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)

def calcular_distancia_ponto(k_range, distorcoes, x , y):
    x0, y0 = k_range[0], distorcoes[0]
    xn, yn = k_range[-1], distorcoes[-1]
    numerador = abs((yn - y0) * x - (xn-x0) * y + xn * y0 - yn * x0)
    return numerador

def achar_k_otimo(k_range, distorcoes):
    comprimento = calcular_comprimento_reta(k_range, distorcoes)
    distancias = []
    
    for i in range(len(distorcoes)):
        x = k_range[i]
        y = distorcoes[i]
        d = calcular_distancia_ponto(k_range, distorcoes, x, y)
        distancias.append(d / comprimento)
    return k_range[distancias.index(np.max(distancias))]

distorcoes = []
k_list = list(K_RANGE)

for i in k_list:
    modelo = KMeans(n_clusters=i, random_state=44, n_init=10).fit(dados_norm)
    distorcoes.append(calcular_distorcao(modelo, dados_norm))

k_otimo = achar_k_otimo(k_list, distorcoes)
print(f'\nK ótimo encontrado: {k_otimo}')
print()


# Treinar modelo com K ótimo

cluster_model = KMeans(n_clusters=k_otimo, random_state=44, n_init=10).fit(dados_norm)

pickle.dump(cluster_model, open(PKL_PATH, 'wb'))


# Descrever os segmentos (recebe modelo treinado -> desnormaliza -> exive valores originais (acho))

def descrever_segmentos(cluster_model, normalizer, colunas_num, colunas_cat_norm, colunas_cat_original):

    all_columns = colunas_num + list(colunas_cat_norm.columns)
    centroides_df = pd.DataFrame(
        cluster_model.cluster_centers_, columns = all_columns
    )

    # Desnomaliza numericos
    centr_num = centroides_df[colunas_num]
    centr_num_orig = normalizer.inverse_transform(centr_num)
    centr_num_orig = pd.DataFrame(centr_num_orig, columns = colunas_num). round(2)

    # Desnormaliza categóricos
    centr_cat = centroides_df[list(colunas_cat_norm.columns)]
    cat_decoded = {}

    for col_original in colunas_cat_original:
        colunas_do_grupo = [c for c in colunas_cat_norm.columns
                            if c.startswith(col_original + '_')]
        
        if not colunas_do_grupo:
            continue

        idx_max = centr_cat[colunas_do_grupo].values.argmax(axis=1)
        prefixo = col_original + '_'
        valores = [colunas_do_grupo[i].replace(prefixo, '', 1) for i in idx_max]
        cat_decoded[col_original] = valores

    centr_cat_orig = pd.DataFrame(cat_decoded)

    resultado = centr_num_orig.join(centr_cat_orig)
    resultado.index.name = 'Cluster'

    print(resultado.to_string())
    print()
    return resultado

segmentos = descrever_segmentos(cluster_model, normalizer, colunas_num, dados_cat_norm, colunas_cat)


# Inferência

def inferir_paciente(paciente_dict, cluster_model, normalizer, colunas_num, colunas_cat_norm):

    dados_paciente_num = {k: v for k, v in paciente_dict.items()
                         if k in colunas_num}
    dados_paciente_cat = {k: v for k, v in paciente_dict.items()
                         if k not in colunas_num}
    
    df_num = pd.DataFrame([dados_paciente_num], columns = colunas_num)
    df_num_norm = normalizer.transform(df_num)
    df_num_norm = pd.DataFrame(df_num_norm, columns = colunas_num)

    # Categóricas
    df_cat = pd.DataFrame([dados_paciente_cat])
    df_cat_norm = pd.get_dummies(df_cat, dtype = int)

    df_cat_alihado = df_cat_norm.reindex(columns=colunas_cat_norm.columns, fill_value=0)

    paciente_norm = pd.concat(
        [df_num_norm.reset_index(drop=True),
         df_cat_alihado.reset_index(drop=True)],
        axis=1
    )

    cluster_id = cluster_model.predict(paciente_norm)[0]
    return cluster_id

paciente_ex = {

    'Age':    22.0,
    'Height': 1.50,
    'Weight': 52.0,
    'FCVC':   2.0,
    'NCP':    3.0,
    'CH2O':   3.0,
    'FAF':    0.0,
    'TUE':    1.0,
    'Gender':                         'Female',
    'family_history_with_overweight': 'yes',
    'FAVC':                           'yes',
    'CAEC':                           'Sometimes',
    'SMOKE':                          'no',
    'SCC':                            'no',
    'CALC':                           'Sometimes',
    'MTRANS':                         'Public_Transportation',

}

cluster_resultado = inferir_paciente(paciente_ex, cluster_model, normalizer, colunas_num, dados_cat_norm)

print(f'Dados do paciente: ')
for k, v in paciente_ex.items():
    print(f'   {k}: {v}')
print()
print(f'Paciente pertencente ao Cluster: {cluster_resultado}')
print()

print(f'Perfil médio cluster {cluster_resultado}')
print(segmentos.iloc[cluster_resultado].to_string())
