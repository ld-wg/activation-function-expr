from activation_functions.CustomGELU import CustomGELU
from activation_functions.CustomReLU import CustomReLU
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Esta versão mantém a mesma arquitetura da rede Holuchi original, com três camadas convolucionais seguidas de pooling e duas camadas densas. 
A única modificação é na camada final, que foi alterada para ter 10 neurônios de saída (em vez de 2), permitindo a classificação em 10 classes para o dataset FashionMNIST.
"""

class Hochuli(nn.Module):
    def __init__(self, activation="relu"):
        """
            activation: string indicando qual função de ativação utilizar ("relu" ou "gelu")
        """
        super(Hochuli, self).__init__()
        self.activation_fn = activation  # define a função de ativação a ser usada
        
        # inicializa os contadores para esta instância da rede
        self.grad_calls = 0
        self.grad_calls_nondiff = 0
        
        # camadas convolucionais (usa padding=1 para preservar dimensões)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # camada de Max Pooling com kernel 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # após três operações de conv + pooling: 32x32 -> 16x16 -> 8x8 -> 4x4
        # vetor achatado terá 64 * 4 * 4 = 1024 elementos
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        # camada final: mapeia 64 features para 10 classes
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def apply_activation(self, x):
        # aplica a função de ativação customizada escolhida, passando a instância da rede para atualizar os contadores
        if self.activation_fn == "relu":
            return CustomReLU.apply(x, self)
        elif self.activation_fn == "gelu":
            return CustomGELU.apply(x, self)
        else:
            raise ValueError("Função de ativação desconhecida. Use 'relu' ou 'gelu'.")

    def forward(self, x):
        # Camada 1: Convolução -> Ativação customizada -> Pooling
        x = self.apply_activation(self.conv1(x))
        x = self.pool(x)
        
        # Camada 2: Convolução -> Ativação customizada -> Pooling
        x = self.apply_activation(self.conv2(x))
        x = self.pool(x)
        
        # Camada 3: Convolução -> Ativação customizada -> Pooling
        x = self.apply_activation(self.conv3(x))
        x = self.pool(x)
        
        # Achata as características para as camadas densas
        x = x.view(x.size(0), -1)
        
        # Camada densa com ativação customizada
        x = self.apply_activation(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
