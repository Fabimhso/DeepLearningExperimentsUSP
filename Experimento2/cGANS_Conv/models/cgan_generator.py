import torch
import torch.nn as nn

class CGANGenerator(nn.Module):
    """
    Gerador convolucional condicional (DC-cGAN)
    Entrada: ruído z + rótulo y
    Saída: imagem 1x28x28
    """
    def __init__(self, z_dim=100, n_classes=10):
        super().__init__()

        # Embedding do rótulo (one-hot projetado)
        self.label_emb = nn.Embedding(n_classes, n_classes)

        # Camada inicial totalmente conectada
        self.fc = nn.Linear(z_dim + n_classes, 128 * 7 * 7)

        # Blocos convolucionais transpostos
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),    # 28x28
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: (B, 100)
        # labels: (B)

        y = self.label_emb(labels)              # (B, 10)
        x = torch.cat([z, y], dim=1)            # (B, 110)

        x = self.fc(x)                          # (B, 128*7*7)
        x = x.view(-1, 128, 7, 7)               # (B, 128, 7, 7)

        return self.deconv(x)                   # (B, 1, 28, 28)
