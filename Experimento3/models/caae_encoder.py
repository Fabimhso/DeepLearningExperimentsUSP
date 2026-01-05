import torch
import torch.nn as nn
import torch.nn.functional as F

class CAAEEncoder(nn.Module):
    """
    Encoder convolucional condicional
    Mapeia imagem -> espaço latente z
    """
    def __init__(self, z_dim=100):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 7x7
            nn.ReLU()
        )

        self.fc = nn.Linear(64 * 7 * 7, z_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
