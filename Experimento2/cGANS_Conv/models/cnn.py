import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """
    CNN simples para classificação do MNIST
    Entrada: imagem 1x28x28
    Saída: logits para 10 classes
    """
    def __init__(self):
        super().__init__()

        # Bloco convolucional 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Entrada: (B, 1, 28, 28)

        x = F.relu(self.conv1(x))        # (B, 32, 28, 28)
        x = F.max_pool2d(x, 2)           # (B, 32, 14, 14)

        x = F.relu(self.conv2(x))        # (B, 64, 14, 14)
        x = F.max_pool2d(x, 2)           # (B, 64, 7, 7)

        x = x.view(x.size(0), -1)        # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)               # Logits
