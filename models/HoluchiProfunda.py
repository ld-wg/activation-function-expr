import activation_functions.CustomGELU as CustomGELU
from activation_functions.CustomReLU import CustomReLU
import torch
import torch.nn as nn
import torch.nn.functional as F

# Em vez de utilizar camadas de max pooling para reduzir as dimensões espaciais, 
# esta versão substitui essas camadas por convoluções com kernel 2×2 e stride 2, 
# que possuem parâmetros treináveis. Essa substituição faz com que o processo de 
# redução das dimensões seja realizado por camadas convolucionais, duplicando 
# efetivamente o número de camadas convolucionais na parte de extração de 
# características (de 3 para 6 camadas), mantendo a parte densa final inalterada. 
# Isso resulta em uma rede mais profunda e com maior número de operações convolucionais, 
# sem uma alteração drástica no número de parâmetros na parte totalmente conectada.

# 3. Hochuli Profunda: Replace max pooling with convolution layers (kernel=2, stride=2).
#    This effectively doubles the convolution layers from 3 to 6.

class HochuliProfunda(nn.Module):
    def __init__(self, activation="relu"):
        """
        Parâmetros:
            activation: string indicando qual função de ativação usar ("relu" ou "gelu")
        """
        super(HochuliProfunda, self).__init__()
        self.activation_fn = activation
        # Inicializa os contadores para esta instância
        self.grad_calls = 0
        self.grad_calls_nondiff = 0

        # Block 1: Convolução + Convolução de downsampling (substitui o pooling)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_down = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        
        # Block 2: Convolução + Convolução de downsampling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2_down = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        
        # Block 3: Convolução + Convolução de downsampling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3_down = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        
        # Dimensões: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)  # 10 classes

    def apply_activation(self, x):
        if self.activation_fn == "relu":
            return CustomReLU.apply(x, self)
        elif self.activation_fn == "gelu":
            return CustomGELU.apply(x, self)
        else:
            raise ValueError("Função de ativação desconhecida. Use 'relu' ou 'gelu'.")

    def forward(self, x):
        # Block 1
        x = self.apply_activation(self.conv1(x))
        x = self.apply_activation(self.conv1_down(x))
        # Block 2
        x = self.apply_activation(self.conv2(x))
        x = self.apply_activation(self.conv2_down(x))
        # Block 3
        x = self.apply_activation(self.conv3(x))
        x = self.apply_activation(self.conv3_down(x))
        
        x = x.view(x.size(0), -1)
        x = self.apply_activation(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# Example usage:
if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 32, 32)
    
    model3 = HochuliProfunda()
    
    print("HochuliProfunda Output:", model3(dummy_input))