"""
Implémentation des Adapters pour Wav2Vec2
Basé sur les papers:
- Houlsby et al. (2019) - Bottleneck Adapters
- Thomas et al. (2022) - Efficient Adapter Transfer for ASR
"""

import torch
import torch.nn as nn
from typing import Optional


class BottleneckAdapter(nn.Module):
    """
    Bottleneck Adapter (Houlsby et al., 2019)
    
    Architecture:
    Input → LayerNorm → Down-projection → Non-linearity → Up-projection → Residual
    
    Args:
        input_dim: Dimension d'entrée (hidden_size du modèle)
        bottleneck_dim: Dimension du goulot d'étranglement (réduction)
        activation: Fonction d'activation (relu, gelu, tanh)
        dropout: Taux de dropout
    """
    def __init__(
        self, 
        input_dim: int,
        bottleneck_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Layer normalization (recommandé par Houlsby)
        self.adapter_layernorm = nn.LayerNorm(input_dim)
        
        # Down-projection (réduction de dimension)
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Activation {activation} non supportée")
        
        # Up-projection (retour à la dimension originale)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # Dropout pour régularisation
        self.dropout = nn.Dropout(dropout)
        
        # Initialisation optimale (proche de l'identité au début)
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation proche de zéro pour démarrage proche de l'identité"""
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec connexion résiduelle
        
        Args:
            hidden_states: Tensor de shape (batch, seq_len, input_dim)
        
        Returns:
            Tensor de même shape avec transformation adapter
        """
        # Sauvegarde pour la connexion résiduelle
        residual = hidden_states
        
        # Adapter transformation
        hidden_states = self.adapter_layernorm(hidden_states)
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.up_project(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Connexion résiduelle (CRUCIAL)
        return residual + hidden_states


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter - Variante optimisée
    Plus rapide que Houlsby car parallèle aux couches FFN
    
    Basé sur He et al. (2021) - "Towards a Unified View of Parameter-Efficient Transfer Learning"
    """
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
        scale: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale  # Facteur d'échelle pour la fusion
        
        # Transformation adapter (sans LayerNorm pour le mode parallèle)
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()
        
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_project.weight, a=0.01)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        
        # Transformation parallèle
        adapter_output = self.down_project(hidden_states)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.dropout(adapter_output)
        adapter_output = self.up_project(adapter_output)
        
        # Fusion avec scaling
        return residual + self.scale * adapter_output


class AdapterConfig:
    """Configuration pour les adapters"""
    def __init__(
        self,
        adapter_type: str = "bottleneck",  # "bottleneck" ou "parallel"
        bottleneck_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1,
        adapter_layers: Optional[list] = [20, 21, 22, 23],  # J'ai la possibilite de specifier les couches sous formr de listes ou de mettre None = tous les layers
        scaling_factor: float = 1.0
    ):
        self.adapter_type = adapter_type
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.dropout = dropout
        self.adapter_layers = adapter_layers
        self.scaling_factor = scaling_factor
    
    def __repr__(self):
        return (f"AdapterConfig(type={self.adapter_type}, "
                f"bottleneck={self.bottleneck_dim}, "
                f"activation={self.activation})")


def create_adapter(
    input_dim: int,
    config: AdapterConfig
) -> nn.Module:
    """
    Factory function pour créer un adapter selon la config
    
    Args:
        input_dim: Dimension d'entrée
        config: Configuration de l'adapter
    
    Returns:
        Module adapter configuré
    """
    if config.adapter_type == "bottleneck":
        return BottleneckAdapter(
            input_dim=input_dim,
            bottleneck_dim=config.bottleneck_dim,
            activation=config.activation,
            dropout=config.dropout
        )
    elif config.adapter_type == "parallel":
        return ParallelAdapter(
            input_dim=input_dim,
            bottleneck_dim=config.bottleneck_dim,
            activation=config.activation,
            dropout=config.dropout,
            scale=config.scaling_factor
        )
    else:
        raise ValueError(f"Type d'adapter inconnu: {config.adapter_type}")


def inject_adapters_into_model(
    model: nn.Module,
    adapter_config: AdapterConfig,
    model_type: str = "wav2vec2"
) -> int:
    """
    Injecte des adapters dans un modèle Wav2Vec2
    
    Selon les bonnes pratiques:
    - Injecte après les couches FFN (Feed-Forward Network)
    - Gèle tous les paramètres sauf les adapters
    - Peut cibler seulement les couches supérieures
    
    Args:
        model: Modèle Wav2Vec2ForCTC
        adapter_config: Configuration des adapters
        model_type: Type de modèle ("wav2vec2", "hubert", etc.)
    
    Returns:
        Nombre de paramètres entraînables ajoutés
    """
    if model_type != "wav2vec2":
        raise NotImplementedError(f"Type {model_type} pas encore supporté")
    
    # Accès aux encoders Wav2Vec2
    encoder = model.wav2vec2.encoder
    num_layers = len(encoder.layers)
    
    # Déterminer quelles couches modifier
    if adapter_config.adapter_layers is None:
        # Par défaut: toutes les couches
        target_layers = list(range(num_layers))
    else:
        target_layers = adapter_config.adapter_layers
    
    print(f"🔧 Injection d'adapters dans les couches: {target_layers}")
    
    total_params = 0
    
    for layer_idx in target_layers:
        layer = encoder.layers[layer_idx]
        hidden_size = layer.feed_forward.intermediate_dense.out_features
        
        # Création de l'adapter
        adapter = create_adapter(hidden_size, adapter_config)
        
        # Injection après le FFN (Feed-Forward Network)
        # On wrap la sortie du FFN avec l'adapter
        original_forward = layer.feed_forward.forward
        
        def make_adapter_forward(original_fn, adapter_module):
            def forward_with_adapter(hidden_states):
                # FFN original
                ffn_output = original_fn(hidden_states)
                # Adapter sur la sortie FFN
                return adapter_module(ffn_output)
            return forward_with_adapter
        
        layer.feed_forward.forward = make_adapter_forward(original_forward, adapter)
        
        # Enregistrer l'adapter comme sous-module pour le sauvegarder
        setattr(layer, f"adapter_{layer_idx}", adapter)
        
        # Compter les paramètres
        adapter_params = sum(p.numel() for p in adapter.parameters())
        total_params += adapter_params
    
    print(f"✅ {total_params:,} paramètres d'adapter ajoutés")
    
    return total_params


def freeze_base_model(model: nn.Module) -> int:
    """
    Gèle tous les paramètres du modèle de base
    
    Args:
        model: Modèle à geler
    
    Returns:
        Nombre de paramètres gelés
    """
    frozen_params = 0
    
    for name, param in model.named_parameters():
        # Geler tout sauf les adapters et le LM head (sortie CTC)
        if "adapter" not in name and "lm_head" not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
    
    print(f"🔒 {frozen_params:,} paramètres gelés (base model)")
    
    return frozen_params


def count_trainable_parameters(model: nn.Module) -> dict:
    """
    Compte les paramètres entraînables vs gelés
    
    Returns:
        Dict avec statistiques détaillées
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": 100 * trainable / total if total > 0 else 0
    }