import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 2**-23


# resumo: a derivada aproximada do gelu é calculada combinando dois termos:
# o primeiro é 0.5*(1 + tanh(u)) e o segundo é 0.5*x*(1 - tanh(u)^2)*(du/dx),
# onde u = sqrt(2/pi) * (x + 0.044715*x^3) e du/dx = sqrt(2/pi) * (1 + 3*0.044715*x^2).
# esse cálculo utiliza a derivada da função tanh (1 - tanh(u)^2) para capturar a não linearidade,
# resultando em uma aproximação eficiente do gradiente do gelu.

class CustomGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, network):
        # salva o objeto network no contexto para uso no backward
        ctx.network = network
        # salva o tensor de entrada para uso posterior na derivada
        ctx.save_for_backward(input)
        # aproximação do GELU:
        # GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
        sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi, device=input.device, dtype=input.dtype))
        # retorna o valor da ativação gelu usando a aproximação com tanh
        return 0.5 * input * (1 + torch.tanh(sqrt_2_over_pi * (input + 0.044715 * input**3)))
    
    @staticmethod
    def backward(ctx, grad_output):
        # recupera o tensor de entrada salvo na fase forward
        input, = ctx.saved_tensors
        # recupera o objeto network salvo para atualizar contadores
        network = ctx.network
        sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi, device=input.device, dtype=input.dtype))

        # cálculo de u = sqrt(2/pi) * (x + 0.044715 * x^3)
        u = sqrt_2_over_pi * (input + 0.044715 * input**3)

        # cálculo de phi = tanh(u), que é a parte não-linear da função gelu
        phi = torch.tanh(u)

        # cálculo da derivada de tanh(u): d(phi)/du = 1 - tanh(u)^2
        dphi_du = 1 - phi**2

        # cálculo da derivada de u em relação a x:
        # du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        du_dx = sqrt_2_over_pi * (1 + 3 * 0.044715 * input**2)

        # cálculo da derivada aproximada do gelu
        # grad = 0.5 * (1 + tanh(u)) + 0.5 * x * (d(phi)/du) * (du/dx)
        grad = 0.5 * (1 + phi) + 0.5 * input * dphi_du * du_dx

        # aplicação da regra da cadeia: gradiente da entrada = grad_output * grad
        grad_input = grad_output * grad
        
        # atualização dos contadores: incrementa o número total de gradientes computados
        num_elements = grad_output.numel()

        network.grad_calls += num_elements
        # gelu é diferenciável em todo o domínio, portanto nenhum ponto é considerado não diferenciável
        network.grad_calls_nondiff += 0
        
        # retorna o gradiente em relação à entrada e None para o objeto network (que não precisa de gradiente)
        return grad_input, None