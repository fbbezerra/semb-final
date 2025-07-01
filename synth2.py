import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Parâmetros (os mesmos de antes) ---
N_AMOSTRAS_POR_CLASSE_BASE = 1500 # Vamos gerar 1500 de cada tipo base
N_AMOSTRAS_POR_CLASSE_COMPOSTA = 1000 # E 1000 de cada tipo composto
TAXA_AMOSTRAGEM = 100
DURACAO_JANELA = 2
N_PONTOS = TAXA_AMOSTRAGEM * DURACAO_JANELA
T = np.linspace(0, DURACAO_JANELA, N_PONTOS, endpoint=False)
GRAVIDADE = -1.0

# --- Funções de Geração de Sinais Puros (Modificadas para serem reutilizáveis) ---

def gerar_sinal_liso_puro():
    acc_x = np.random.randn(N_PONTOS) * 0.03
    acc_y = np.random.randn(N_PONTOS) * 0.03
    acc_z = GRAVIDADE + (np.random.randn(N_PONTOS) * 0.04)
    gyro_x = np.random.randn(N_PONTOS) * 2.5
    gyro_y = np.random.randn(N_PONTOS) * 2.5
    gyro_z = np.random.randn(N_PONTOS) * 2.5
    return np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1)

def gerar_sinal_irregular_puro():
    sinal_vib_z = 0.4 * np.sin(2 * np.pi * 18 * T) + 0.2 * np.sin(2 * np.pi * 25 * T)
    sinal_vib_y = 0.3 * np.sin(2 * np.pi * 15 * T)
    acc_x = np.random.randn(N_PONTOS) * 0.1
    acc_y = sinal_vib_y + (np.random.randn(N_PONTOS) * 0.15)
    acc_z = GRAVIDADE + sinal_vib_z + (np.random.randn(N_PONTOS) * 0.2)
    gyro_x = 40 * np.sin(2 * np.pi * 10 * T) + np.random.randn(N_PONTOS) * 15
    gyro_y = 25 * np.sin(2 * np.pi * 12 * T) + np.random.randn(N_PONTOS) * 15
    gyro_z = 30 * np.sin(2 * np.pi * 8 * T) + np.random.randn(N_PONTOS) * 15
    return np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1)

def gerar_impulso_buraco():
    """Gera apenas o impulso do buraco, para ser adicionado a outros sinais."""
    impulso = np.zeros((N_PONTOS, 6))
    posicao_impacto = np.random.uniform(DURACAO_JANELA * 0.3, DURACAO_JANELA * 0.7)
    largura_impacto_acc = 0.025
    largura_impacto_gyro = 0.022
    impulso[:, 2] += 3.5 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_acc**2)) # Z Acc
    impulso[:, 0] -= 2.0 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_acc**2)) # X Acc
    impulso[:, 1] += 1.5 * np.exp(-((T - (posicao_impacto+0.005))**2) / (2 * largura_impacto_acc**2)) # Y Acc
    impulso[:, 3] += 450 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_gyro**2)) # X Gyro
    impulso[:, 4] -= 300 * np.exp(-((T - (posicao_impacto+0.005))**2) / (2 * largura_impacto_gyro**2)) # Y Gyro
    impulso[:, 5] += 100 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_gyro**2)) # Z Gyro
    return impulso

def gerar_sinal_buraco_puro():
    """Buraco em terreno liso."""
    return gerar_sinal_liso_puro() + gerar_impulso_buraco()

# --- Novas Funções para Sinais Compostos ---

def gerar_irregular_com_buraco():
    """Gera um sinal de terreno irregular com um buraco sobreposto."""
    sinal_irregular = gerar_sinal_irregular_puro()
    impulso = gerar_impulso_buraco()
    return sinal_irregular + impulso

def gerar_liso_para_irregular():
    """Gera uma transição de terreno liso para irregular."""
    ponto_transicao = N_PONTOS // 2
    metade_lisa = gerar_sinal_liso_puro()[:ponto_transicao, :]
    metade_irregular = gerar_sinal_irregular_puro()[ponto_transicao:, :]
    return np.concatenate((metade_lisa, metade_irregular), axis=0)
    
# --- Geração do Novo Dataset ---

print("Gerando dataset realista com sinais puros e compostos...")
lista_de_dataframes = []
sample_id_counter = 0

# Define as funções e seus respectivos rótulos de acordo com a regra de severidade
mapeamento_geradores = {
    "liso": (gerar_sinal_liso_puro, N_AMOSTRAS_POR_CLASSE_BASE),
    "irregular": (gerar_sinal_irregular_puro, N_AMOSTRAS_POR_CLASSE_BASE),
    "buraco": (gerar_sinal_buraco_puro, N_AMOSTRAS_POR_CLASSE_BASE),
    "buraco_em_irregular": (gerar_irregular_com_buraco, N_AMOSTRAS_POR_CLASSE_COMPOSTA),
    "transicao_liso_irregular": (gerar_liso_para_irregular, N_AMOSTRAS_POR_CLASSE_COMPOSTA)
}

mapeamento_rotulos = {
    "liso": "liso",
    "irregular": "irregular",
    "buraco": "buraco",
    "buraco_em_irregular": "buraco", # Rótulo é a classe mais severa
    "transicao_liso_irregular": "irregular" # Rótulo é a classe mais severa
}

for nome_gerador, (func_geradora, n_amostras) in tqdm(mapeamento_geradores.items(), desc="Tipos de Sinal"):
    for _ in range(n_amostras):
        dados_6_eixos = func_geradora()
        df_amostra = pd.DataFrame(dados_6_eixos, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        df_amostra['sample_id'] = sample_id_counter
        df_amostra['label'] = mapeamento_rotulos[nome_gerador]
        df_amostra['timestamp'] = T
        lista_de_dataframes.append(df_amostra)
        sample_id_counter += 1

print("Concatenando dataframes...")
df_final = pd.concat(lista_de_dataframes, ignore_index=True)
df_final = df_final[['sample_id', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']]

print("Salvando arquivo CSV...")
df_final.to_csv('pavement_simulation_data_realista.csv', index=False)

print("\nArquivo 'pavement_simulation_data_realista.csv' gerado com sucesso!")
print(f"Total de amostras: {df_final['sample_id'].nunique()}")
print(f"Distribuição das classes no novo dataset:\n{df_final.groupby('sample_id')['label'].first().value_counts()}")
