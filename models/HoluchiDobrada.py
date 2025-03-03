import activation_functions.CustomGELU as CustomGELU
from activation_functions.CustomReLU import CustomReLU
import torch
import torch.nn as nn
import torch.nn.functional as F

# Nesta variação, a estrutura da rede permanece inalterada quanto ao número de camadas, 
# mas o número de filtros em cada camada convolucional é dobrado. Ou seja, se a rede original 
# tinha 32 filtros na primeira camada, agora passa a ter 64; se tinha 64 na segunda e terceira, 
# passa a ter 128 em cada. Essa mudança aumenta significativamente o número de parâmetros 
# e a capacidade de extração de características da rede.

class HochuliDobrada(nn.Module):
    def __init__(self, activation="relu"):
        """
        Parâmetros:
            activation: string indicando qual função de ativação usar ("relu" ou "gelu")
        """
        super(HochuliDobrada, self).__init__()
        self.activation_fn = activation
        # Inicializa os contadores para esta instância
        self.grad_calls = 0
        self.grad_calls_nondiff = 0

        # Dobrando o número de filtros:
        # conv1: 32 -> 64; conv2: 64 -> 128; conv3: 64 -> 128.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # tamanho do vetor achatado: 128 canais * 4 * 4 = 2048.
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)  # 10 classes

    def apply_activation(self, x):
        if self.activation_fn == "relu":
            return CustomReLU.apply(x, self)
        elif self.activation_fn == "gelu":
            return CustomGELU.apply(x, self)
        else:
            raise ValueError("Função de ativação desconhecida. Use 'relu' ou 'gelu'.")

    def forward(self, x):
        x = self.apply_activation(self.conv1(x))
        x = self.pool(x)
        
        x = self.apply_activation(self.conv2(x))
        x = self.pool(x)
        
        x = self.apply_activation(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.apply_activation(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x