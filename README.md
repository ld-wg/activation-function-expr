# Experimento com Funções de Ativação

Este experimento compara duas funções de ativação: **ReLU** e **GELU**, utilizando a **CNN Holuchi** modificada para saída em 10 classes.

## Parâmetros do Experimento

- **num_epochs** = 50
- **batch_size** = 64
- **learning_rate** = 0.001
- **momentum** = 0.9
- **num_runs** = 3 (cada experimento é executado 3 vezes com modelos independentes)

## Estrutura do Projeto

- **/model:** Contém os modelos gerados automaticamente ao rodar `main.py`.
- **/results:** Diretório onde os gráficos são salvos.
- **/saved_models:** Armazena os modelos treinados.
- **main.py:** Treina a CNN e gera `results.json`.
- **Analise.py:** Gera gráficos de análise a partir de `results.json`.
- **Results.py:** Define a estrutura de dados do `results.json`.

## Tutorial de Execução

1. **Ativar ambiente virtual:**

   ```bash
   source .env/bin/activate
   ```

2. **Instalar dependências:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Executar treinamento:**

   ```bash
   python main.py
   ```

4. **Analisar resultados:**
   ```bash
   python Analise.py
   ```

Os resultados serão armazenados no `results.json` e os gráficos no diretório `/results`.

## Divisão do Dataset

O conjunto de dados original contém **70.000 imagens**, distribuídas da seguinte forma:

```python
# Combina os conjuntos em um único dataset
full_dataset = ConcatDataset([train_dataset, test_dataset])
total = len(full_dataset)  # 70k imagens

# Calcula os tamanhos de cada divisão:
train_size = int(0.6 * total)  # 60% de 70k = 42k imagens
val_size   = int(0.25 * total) # 25% de 70k = 17.5k imagens
test_size  = total - train_size - val_size  # 10.5k imagens
```

## Estrutura da CNN Holuchi

### Camadas Convolucionais:

- Camada 1: 32 filtros, kernel 3×3.
- Camada 2: 64 filtros, kernel 3×3.
- Camada 3: 64 filtros, kernel 3×3.
- ReLU após cada convolução + Max Pooling 2×2.

### Camadas Densas:

- Camada densa de 64 neurônios + ReLU.
- Camada final mapeando para 10 classes + Softmax.

## Implementação da Ativação GELU Customizada

A classe `CustomGELU` utiliza `torch.autograd.Function` para monitoramento das chamadas de gradiente.

### Cálculo da Ativação GELU

A ativação é aproximada por:

```math
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

### Cálculo do Gradiente

A derivada do GELU é calculada como:

```math
grad ≈ 0.5 × (1 + tanh(u)) + 0.5 × x × (1 - tanh²(u)) × du/dx
```

onde:

```math
u = √(2/π) × (x + 0.044715 × x³)
du/dx = √(2/π) × (1 + 3 × 0.044715 × x²)
```
