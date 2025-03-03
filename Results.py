from dataclasses import dataclass, field
from typing import List

@dataclass
class EpochResult:
    epoch: int                 # Número da época
    train_loss: float          # Perda de treino nesta época
    val_loss: float            # Perda de validação nesta época
    epoch_time: float          # Tempo gasto na época (em segundos)
    grad_calls: int            # Total de chamadas de gradiente nesta época
    grad_calls_nondiff: int    # Chamadas em pontos próximos à não diferenciabilidade

@dataclass
class RunResult:
    epoch_info: List[EpochResult] = field(default_factory=list)  # Lista com os resultados de cada época
    test_loss: float = 0.0              # Perda no conjunto de teste (ao final da execução)
    test_accuracy: float = 0.0          # Acurácia no conjunto de teste (ao final da execução)

@dataclass
class ConfigResult:
    config_name: str                   # Nome da configuração (por exemplo, "Holuchi_relu")
    runs: List[RunResult] = field(default_factory=list)  # Lista com os resultados de cada execução (run)