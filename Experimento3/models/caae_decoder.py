import torch
import torch.nn as nn

class CAAEDecoder(nn.Module):
    """
    Decoder convolucional condicional
    Recebe z + rótulo e reconstrói imagem
    """
    def __init__(self, z_dim=100, n_classes=10):
        super().__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.fc = nn.Linear(z_dim + n_classes, 64 * 7 * 7)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        y = self.label_emb(labels)
        x = torch.cat([z, y], dim=1)
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        return self.deconv(x)
