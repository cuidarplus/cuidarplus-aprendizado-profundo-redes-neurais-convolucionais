"""Arquitetura da rede neural MLP (PyTorch) para classificação binária Furtado x Roubado."""
from typing import Sequence

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Multilayer Perceptron parametrizável.

    Cada camada oculta é um bloco: Linear -> [BatchNorm1d] -> Ativação -> [Dropout].
    A saída é um único logit (sem sigmoid), para uso com `nn.BCEWithLogitsLoss`.

    Args:
        input_dim: número de features de entrada.
        hidden_dims: neurônios de cada camada oculta; o tamanho da sequência
            define a profundidade. Ex.: (128, 64) = duas camadas ocultas.
        dropout: taxa única para todas as camadas ou uma taxa por camada.
        activation: "relu", "leaky_relu", "gelu" ou "tanh".
        batch_norm: aplica BatchNorm1d após cada camada linear oculta.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float | Sequence[float] = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        dropouts = [dropout] * len(hidden_dims) if isinstance(dropout, (int, float)) else list(dropout)
        if len(dropouts) != len(hidden_dims):
            raise ValueError("`dropout` deve ser um float ou ter uma taxa por camada oculta.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Ativação desconhecida: {activation}. Opções: {list(ACTIVATIONS)}")

        # Guardado para reconstruir a rede ao carregar os pesos (ver `from_config`).
        self.config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropouts,
            "activation": activation,
            "batch_norm": batch_norm,
        }

        layers: list[nn.Module] = []
        in_features = input_dim
        for out_features, p in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(ACTIVATIONS[activation]())
            if p > 0:
                layers.append(nn.Dropout(p))
            in_features = out_features
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna os logits com shape (batch,)."""
        return self.output(self.hidden(x)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidade da classe positiva (Roubado = 1), em modo de avaliação."""
        self.eval()
        return torch.sigmoid(self(x))

    @classmethod
    def from_config(cls, config: dict) -> "MLP":
        return cls(**config)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
