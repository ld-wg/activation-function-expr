import json
import os
from dataclasses import asdict
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# carrega os resultados salvos em JSON

json_path = "results.json"
with open(json_path, "r") as f:
    results_json = json.load(f)

# filtra os resultados para a rede Holuchi para as duas funções de ativação

configs = { res["config_name"]: res for res in results_json if res["config_name"].startswith("Holuchi_") }

# =============================================================================
# Função 1: Gráfico da Proporção de Não Diferenciabilidade por Época (Holuchi com ReLU)
# =============================================================================

def plot_nondiff_proportion(configs: dict, save_dir: str):
    # seleciona configs de Holuchi com ReLU
    relu_config = configs.get("Holuchi_relu")
    if relu_config is None:
        raise ValueError("Não foram encontrados resultados para 'Holuchi_relu'.")

    num_epochs = len(relu_config["runs"][0]["epoch_info"])
    proportions_per_epoch = []

    for epoch_idx in range(num_epochs):
        epoch_props = []
        for run in relu_config["runs"]:
            epoch_info = run["epoch_info"][epoch_idx]
            # evita divisao por zero
            if epoch_info["grad_calls"] > 0:
                prop = epoch_info["grad_calls_nondiff"] / epoch_info["grad_calls"]
            else:
                prop = 0.0
            epoch_props.append(prop)
        avg_prop = np.mean(epoch_props)
        proportions_per_epoch.append(avg_prop)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), proportions_per_epoch, marker="o", linestyle="-", color="tab:blue")
    plt.xlabel("Época")
    plt.ylabel("Proporção de Não Diferenciabilidade")
    plt.title("Proporção de Não Diferenciabilidade por Época (Holuchi com ReLU)")
    plt.grid(True)
    plt.tight_layout()
    
    # salva o gráfico
    save_path = os.path.join(save_dir, "nondiff_proportion.png")
    plt.savefig(save_path)
    print(f"Gráfico 'nondiff_proportion.png' salvo em {save_dir}")
    plt.close()

# =============================================================================
# Função 2: Gráfico das Curvas de Perda de Validação para Holuchi com ReLU vs GELU
# =============================================================================

def average_validation_loss(config: dict) -> List[float]:
    runs = config["runs"]
    n_epochs = len(runs[0]["epoch_info"])
    avg_losses = []
    for epoch_idx in range(n_epochs):
        losses = [run["epoch_info"][epoch_idx]["val_loss"] for run in runs]
        avg_losses.append(np.mean(losses))
    return avg_losses

def plot_val_loss_curves(configs: dict, save_dir: str):
    # obtemos as curvas para ReLU e GELU
    relu_config = configs.get("Holuchi_relu")
    gelu_config = configs.get("Holuchi_gelu")
    if relu_config is None or gelu_config is None:
        raise ValueError("Resultados para 'Holuchi_relu' e/ou 'Holuchi_gelu' não foram encontrados.")

    # ambas as redes devem possuir o mesmo número de épocas
    num_epochs = len(relu_config["runs"][0]["epoch_info"])
    relu_val_loss = average_validation_loss(relu_config)
    gelu_val_loss = average_validation_loss(gelu_config)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), relu_val_loss, marker="o", linestyle="-", label="ReLU", color="tab:blue")
    plt.plot(range(1, num_epochs + 1), gelu_val_loss, marker="s", linestyle="--", label="GELU", color="tab:orange")
    plt.xlabel("Época")
    plt.ylabel("Perda de Validação")
    plt.title("Curva de Perda de Validação (Holuchi: ReLU vs GELU)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # salva o grafico
    save_path = os.path.join(save_dir, "val_loss_curves.png")
    plt.savefig(save_path)
    print(f"Gráfico 'val_loss_curves.png' salvo em {save_dir}")
    plt.close()

# =============================================================================
# Função 3: Tabela Comparativa com Métricas
# =============================================================================

def compute_metrics(config: dict) -> dict:
    runs = config["runs"]
    n_epochs = len(runs[0]["epoch_info"])
    
    avg_val_losses = []
    avg_epoch_times = []
    for epoch_idx in range(n_epochs):
        losses = [run["epoch_info"][epoch_idx]["val_loss"] for run in runs]
        times = [run["epoch_info"][epoch_idx]["epoch_time"] for run in runs]
        avg_val_losses.append(np.mean(losses))
        avg_epoch_times.append(np.mean(times))
    
    best_epoch = int(np.argmin(avg_val_losses)) + 1  # Soma 1 para converter em contagem de época
    avg_test_accuracy = np.mean([run["test_accuracy"] for run in runs])
    avg_epoch_time = np.mean(avg_epoch_times)
    
    # não medimos o "tempo médio de precisão"
    avg_test_time = "N/A"
    
    return {
        "Melhor Época": best_epoch,
        "Acurácia (Teste)": avg_test_accuracy,
        "Tempo Médio por Época (s)": avg_epoch_time,
        "Tempo Médio de Precisão": avg_test_time,
    }

def create_comparative_table(configs: dict, save_dir: str):
    metrics_relu = compute_metrics(configs.get("Holuchi_relu"))
    metrics_gelu = compute_metrics(configs.get("Holuchi_gelu"))

    table_data = [
        {"Configuração": "Não Diferenciável (ReLU)", **metrics_relu},
        {"Configuração": "Diferenciável (GELU)", **metrics_gelu},
    ]
    
    df = pd.DataFrame(table_data)
    print("\nTabela Comparativa:")
    print(df.to_string(index=False))
    
    # cria uma figura com Matplotlib para exibir a tabela
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # salva a figura como uma imagem
    table_path = os.path.join(save_dir, "comparative_table.png")
    plt.tight_layout()
    plt.savefig(table_path)
    print(f"Tabela comparativa salva como imagem em: {table_path}")
    plt.close()

# =============================================================================
# Chama as sub-rotinas de análise e salva os gráficos
# =============================================================================

def main():
    # cria o diretório para salvar os gráficos
    save_dir = "./results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # gera gráficos e tabela
    plot_nondiff_proportion(configs, save_dir)
    plot_val_loss_curves(configs, save_dir)
    create_comparative_table(configs, save_dir)

if __name__ == '__main__':
    main()