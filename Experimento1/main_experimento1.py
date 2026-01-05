# =========================================================
# main_experimento1.py
# Script principal experimento 1 e
# 1) CARREGA OS DADOS
# 2) TREINA UMA CNN SOBRE DAOOS REAIS
# 3) TESTA A CNN SOBRE DADOS REAIS
# =========================================================

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Importa modelos
from models import (
    CNNClassifier,
    ConditionalGenerator,
    ConditionalDiscriminator
)

# Importa rotinas de treinamento
from training import (
    train_classifier,
    train_cgan,
    plot_confusion,
    compute_statistics,
    calculate_fid
)

# =========================================================
# CONFIGURAÇÃO
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# =========================================================
# DATASET MNIST
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# =========================================================
# TREINAMENTO DE UMA CNN COM DADOS REAIS
# =========================================================
cnn_real = CNNClassifier().to(device)
train_classifier(cnn_real, train_loader, device)


# Faz o gráfico da matriz de confusão
plot_confusion(cnn_real, test_loader, device,
               "CNN treinada com dados reais")



