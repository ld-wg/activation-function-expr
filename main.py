from dataclasses import asdict
import json
from typing import List
from models.Holuchi import Hochuli
from models.HoluchiDobrada import HochuliDobrada
from models.HoluchiProfunda import HochuliProfunda
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import time
import os

from Results import ConfigResult, EpochResult, RunResult

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    # reinicia os contadores para a epoca
    model.grad_calls = 0
    model.grad_calls_nondiff = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # zera o gradiente acumulado de outras epocas
        optimizer.zero_grad()
        # forward pass
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_time, model.grad_calls, model.grad_calls_nondiff

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
    epoch_loss = running_loss / len(val_loader.dataset)
    return epoch_loss

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    running_loss = 0.0
    # desativa calculo de gradientes, pois nao realizamos backpropagation durante validacao
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            # target.view_as(pred) - reformata rolutos de target para o formato de pred
            # pred.eq(target) - cria vetor booleano com 1 para correto 0 para incorreto
            # .sum().item() soma valores true (conta como 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# =============================================================================
# Main: treina, avalia e salva os modelos
# =============================================================================

def main():
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    momentum = 0.9
    num_runs = 3  # executa cada experimento 3 vezes (3 instancias/modelos independentes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # preparacao do dataset FashionMNIST, imagens originais (28x28) serao redimensionadas para 32x32 e convertidas para 3 canais.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # carrega datasets de treino e teste
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # combina os conjuntos em um único dataset
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    total = len(full_dataset)  # 70k imagens

    
    # calcula os tamanhos de cada divisão:
    train_size = int(0.6 * total)  # 60% de 70k = 42k imagens
    val_size   = int(0.25 * total) # 25% de 70k = 17.5k imagens
    test_size  = total - train_size - val_size  # 70k - 42k - 17.5k = 10.5k imagens
    
    # divide dataset
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # cria diretorio para salvar os modelos, se nao existir
    save_dir = "./saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # dicionario de classes de modelo a serem testadas, comente modelos aqui para remove-lo do experimento
    model_configs = {
        "Holuchi": Hochuli,
        # "HochuliDobrada": HochuliDobrada,
        # "HochuliProfunda": HochuliProfunda
    }
    
    activations = ["relu", "gelu"]
    results: List[ConfigResult] = []  # dicionario que mapeia o nome da configuração para a classe ConfigResult
    
    # para cada modelo e para cada funcao de ativacao, treina ${num_runs} instâncias
    for model_name, ModelClass in model_configs.items():
        for act in activations:
            config_name = f"{model_name}_{act}"

            # cria um objeto ConfigResult para a tupla model_activation
            config_result = ConfigResult(config_name=config_name)

            for run in range(num_runs):
                # fixa a semente
                seed = 42 + run
                torch.manual_seed(seed)
                
                # instancia o modelo com a ativação escolhida e move para o dispositivo
                model = ModelClass(activation=act).to(device)
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
                criterion = nn.CrossEntropyLoss()
                
                # cria um objeto RunResult para a execucao
                run_result = RunResult()
                print(f"\nTreinando {config_name} - Execução {run+1}")

                for epoch in range(num_epochs):
                    train_loss, epoch_time, grad_calls, grad_calls_nondiff = train_model(model, train_loader, optimizer, criterion, device)
                    val_loss = validate_model(model, val_loader, criterion, device)
                    
                    # cria um objeto EpochResult para os dados desta epoca
                    epoch_result = EpochResult(
                        epoch=epoch+1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        epoch_time=epoch_time,
                        grad_calls=grad_calls,
                        grad_calls_nondiff=grad_calls_nondiff
                    )
                    run_result.epoch_info.append(epoch_result)

                    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                          f"Time={epoch_time:.2f}s, grad_calls={grad_calls}, nondiff_calls={grad_calls_nondiff}")
                
                test_loss, test_acc = test_model(model, test_loader, criterion, device)
                run_result.test_loss = test_loss
                run_result.test_accuracy = test_acc
                config_result.runs.append(run_result)
                print(f"Resultado Final - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
                
                # salva a instancia do modelo treinado (com seus parâmetros e contadores)
                save_path = os.path.join(save_dir, f"{config_name}_run{run+1}.pt")
                torch.save(model, save_path)
                print(f"Modelo salvo em: {save_path}")
            
            results.append(config_result)

    # exibe resultados agregados para cada config
    print("\nResultados Agregados:")
    for config_result in results:
        avg_test_loss = sum(run.test_loss for run in config_result.runs) / len(config_result.runs)
        avg_test_acc = sum(run.test_accuracy for run in config_result.runs) / len(config_result.runs)
        print(f"{config_result.config_name}: Avg Test Loss={avg_test_loss:.4f}, Avg Test Accuracy={avg_test_acc:.4f}")

    # salva a lista de resultados em um arquivo JSON
    results_json = [asdict(config) for config in results]
    json_path = "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"\nResultados salvos em {json_path}")

if __name__ == '__main__':
    main()