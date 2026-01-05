import torch
import torch.nn as nn
import torch.nn.functional as F

class CGANDiscriminator(nn.Module):
    """
    Discriminador convolucional condicional
    Entrada: imagem + mapa de classe
    Saída: probabilidade real/fake
    """
    def __init__(self, n_classes=10):
        super().__init__()

        # Embedding do rótulo como mapa espacial
        self.label_emb = nn.Embedding(n_classes, 28 * 28)

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x, labels):
        # x: (B, 1, 28, 28)

        y = self.label_emb(labels)               # (B, 784)
        y = y.view(-1, 1, 28, 28)                # (B, 1, 28, 28)

        x = torch.cat([x, y], dim=1)             # (B, 2, 28, 28)

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        return torch.sigmoid(self.fc(x))
