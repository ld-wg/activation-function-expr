import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 2**-23

class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, network):
        ctx.network = network  # armazena a referência à rede para atualizar os contadores
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        network = ctx.network
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0  # derivada da ReLU: 1 para x > 0; 0 para x <= 0
        
        # atualiza contadores
        num_elements = grad_output.numel()
        network.grad_calls += num_elements
        
        # considera-se não diferenciável pontos próximos de 0: |x| <= EPSILON
        nondiff_mask = input.abs() <= EPSILON
        count_nondiff = nondiff_mask.sum().item()
        network.grad_calls_nondiff += count_nondiff
        
        return grad_input, None