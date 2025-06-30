import numpy as np
import pandas as pd
from tqdm import tqdm # Para uma barra de progresso visual

# --- Parâmetros do Dataset ---
N_AMOSTRAS_POR_CLASSE = 2000
TAXA_AMOSTRAGEM = 100  # Hz
DURACAO_JANELA = 2     # Segundos
N_PONTOS = TAXA_AMOSTRAGEM * DURACAO_JANELA
T = np.linspace(0, DURACAO_JANELA, N_PONTOS, endpoint=False)
GRAVIDADE = -1.0

# --- Funções de Geração de Sinais para 6 Eixos ---

def gerar_sinal_liso():
    """Gera um sinal com ruído de baixa amplitude, simulando um passeio suave."""
    acc_x = np.random.randn(N_PONTOS) * 0.03
    acc_y = np.random.randn(N_PONTOS) * 0.03
    acc_z = GRAVIDADE + (np.random.randn(N_PONTOS) * 0.04)

    gyro_x = np.random.randn(N_PONTOS) * 2.5
    gyro_y = np.random.randn(N_PONTOS) * 2.5
    gyro_z = np.random.randn(N_PONTOS) * 2.5

    return np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1)

def gerar_sinal_irregular():
    """Gera vibração contínua em múltiplos eixos."""
    # Vibração base
    sinal_vib_z = 0.4 * np.sin(2 * np.pi * 18 * T) + 0.2 * np.sin(2 * np.pi * 25 * T)
    sinal_vib_y = 0.3 * np.sin(2 * np.pi * 15 * T)

    acc_x = np.random.randn(N_PONTOS) * 0.1
    acc_y = sinal_vib_y + (np.random.randn(N_PONTOS) * 0.15)
    acc_z = GRAVIDADE + sinal_vib_z + (np.random.randn(N_PONTOS) * 0.2)

    gyro_x = 40 * np.sin(2 * np.pi * 10 * T) + np.random.randn(N_PONTOS) * 15
    gyro_y = 25 * np.sin(2 * np.pi * 12 * T) + np.random.randn(N_PONTOS) * 15
    gyro_z = 30 * np.sin(2 * np.pi * 8 * T) + np.random.randn(N_PONTOS) * 15

    return np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1)

def gerar_sinal_buraco():
    """Gera um forte impacto correlacionado entre acelerômetro e giroscópio."""
    # Começa com um sinal base liso
    sinal_base = gerar_sinal_liso()

    posicao_impacto = np.random.uniform(DURACAO_JANELA * 0.3, DURACAO_JANELA * 0.7)
    largura_impacto_acc = 0.025
    largura_impacto_gyro = 0.022

    # Impulsos correlacionados
    # Accel: Z(vertical) tem o maior impacto, X(frente) tem desaceleração, Y(lado) tem balanço
    sinal_base[:, 2] += 3.5 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_acc**2))
    sinal_base[:, 0] -= 2.0 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_acc**2))
    sinal_base[:, 1] += 1.5 * np.exp(-((T - (posicao_impacto+0.005))**2) / (2 * largura_impacto_acc**2))

    # Gyro: X(pitch) tem a maior rotação, seguido por Y(roll)
    sinal_base[:, 3] += 450 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_gyro**2))
    sinal_base[:, 4] -= 300 * np.exp(-((T - (posicao_impacto+0.005))**2) / (2 * largura_impacto_gyro**2))
    sinal_base[:, 5] += 100 * np.exp(-((T - posicao_impacto)**2) / (2 * largura_impacto_gyro**2))

    return sinal_base

# --- Geração do DataFrame ---

print("Gerando dados sintéticos. Isso pode levar um minuto...")

lista_de_dataframes = []
sample_id_counter = 0

mapeamento_classes = {
    0: gerar_sinal_liso,
    1: gerar_sinal_irregular,
    2: gerar_sinal_buraco
}

labels_nomes = {
    0: "liso",
    1: "irregular",
    2: "buraco"
}

# Usando tqdm para a barra de progresso
for classe_id, func_geradora in tqdm(mapeamento_classes.items(), desc="Classes"):
    for i in range(N_AMOSTRAS_POR_CLASSE):
        dados_6_eixos = func_geradora()

        df_amostra = pd.DataFrame(dados_6_eixos, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        df_amostra['sample_id'] = sample_id_counter
        df_amostra['label'] = labels_nomes[classe_id]

        # Adiciona uma coluna de timestamp dentro de cada amostra
        df_amostra['timestamp'] = T

        lista_de_dataframes.append(df_amostra)
        sample_id_counter += 1

print("Concatenando dataframes...")
df_final = pd.concat(lista_de_dataframes, ignore_index=True)

# Reorganiza as colunas para melhor leitura
df_final = df_final[['sample_id', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']]

print("Salvando arquivo CSV...")
df_final.to_csv('pavement_simulation_data.csv', index=False)

print("\nArquivo 'pavement_simulation_data.csv' gerado com sucesso!")
print(f"Total de amostras: {df_final['sample_id'].nunique()}")
print(f"Total de linhas: {len(df_final)}")
print("\nVisualização das 5 primeiras linhas do arquivo:")
print(df_final.head())
