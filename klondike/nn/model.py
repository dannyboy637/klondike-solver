import torch
import torch.nn as nn
import torch.nn.functional as F
from klondike.encoding import OBSERVATION_DIM
from klondike.actions import ACTION_SPACE_SIZE


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and skip connection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Fixup initialization (KataGo: drop batch norm, 1.6-1.8x speedup)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.relu(x + residual)


class KlondikeNet(nn.Module):
    """
    MLP-ResNet for Klondike Solitaire.

    Architecture:
    - Input projection: Linear(757, 512) -> ReLU
    - 4 residual blocks (512 hidden, Fixup init, no batch norm)
    - Policy head: Linear(512, 260) -> masked log_softmax
    - Value head: Linear(512, 256) -> ReLU -> Linear(256, 1) -> sigmoid
    - Auxiliary heads: foundation count, face-down count (dormant in Phase 3)

    ~2.75M parameters at default settings.
    """

    def __init__(
        self,
        input_dim: int = OBSERVATION_DIM,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        action_dim: int = ACTION_SPACE_SIZE,
        use_auxiliary: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual tower
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)
        ])

        # Policy head
        self.policy_fc = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value_fc1 = nn.Linear(hidden_dim, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Auxiliary heads (KataGo insight — sharper gradients, dormant in Phase 3)
        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            self.aux_foundation = nn.Linear(hidden_dim, 1)
            self.aux_facedown = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            obs: (batch, input_dim) observation vectors
            legal_mask: (batch, action_dim) binary mask of legal actions

        Returns dict with:
            "policy": (batch, action_dim) log-probabilities (masked)
            "value": (batch, 1) win probability [0, 1]
            "aux_foundation": (batch, 1) predicted foundation count [optional]
            "aux_facedown": (batch, 1) predicted face-down count [optional]
        """
        # Shared trunk
        x = F.relu(self.input_proj(obs))
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)

        # Policy head
        logits = self.policy_fc(x)
        if legal_mask is not None:
            logits = logits + (1 - legal_mask) * (-1e9)
        log_policy = F.log_softmax(logits, dim=-1)

        # Value head
        v = F.relu(self.value_fc1(x))
        v = torch.sigmoid(self.value_fc2(v))

        result = {"policy": log_policy, "value": v}

        if self.use_auxiliary:
            result["aux_foundation"] = self.aux_foundation(x)
            result["aux_facedown"] = self.aux_facedown(x)

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
