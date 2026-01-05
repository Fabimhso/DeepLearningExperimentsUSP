import torch
import torch.nn as nn

class CAAEDiscriminator(nn.Module):
    """
    Discriminador no espaço latente
    Força z ~ N(0, I)
    """
    def __init__(self, z_dim=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
