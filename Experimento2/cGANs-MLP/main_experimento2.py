# =========================================================
# main_experimento12.py
# Script principal experimento 1 e
# 1) CARREGA OS DADOS
# 2) TREINA UMA CNN SOBRE DAOOS REAIS
# 3) TREINA UMA CGANS
# 4) GERA DADOS FAKES
# 5) TREINA UMA CNN COM DADOS REAIS E FAKES
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
plot_confusion(cnn_real, test_loader, device,
               "CNN treinada com dados reais")

# =========================================================
# TREINAMENTO DE UMA cGAN convolucional
# =========================================================
G = ConditionalGenerator().to(device)
D = ConditionalDiscriminator().to(device)

train_cgan(G, D, train_loader, device)

# =========================================================
# GERAÇÃO DE DADOS FAKE
# =========================================================
def generate_fake_dataset(G, n_samples=60000):
    z = torch.randn(n_samples, 100, device=device)
    y = torch.randint(0, 10, (n_samples,), device=device)
    x = G(z, y).cpu()
    return TensorDataset(x, y.cpu())

fake_dataset = generate_fake_dataset(G)
fake_loader = DataLoader(fake_dataset, batch_size=128, shuffle=True)

# =========================================================
# TREINAMENTO DE UMA CNN COM DADOS FAKE
# =========================================================
cnn_fake = CNNClassifier().to(device)
train_classifier(cnn_fake, fake_loader, device)

#FAZ O GRÁFICO DA MATRIZ DE CONFUSÃO
plot_confusion(cnn_fake, test_loader, device,
               "CNN treinada com dados fake (cGAN)")

# =========================================================
# CALCULA O FID PARA VERIFICAR AS IMAGENS GERADAS
# =========================================================
feature_extractor = cnn_real.features

mu_real, sigma_real = compute_statistics(
    train_loader, feature_extractor, device
)

mu_fake, sigma_fake = compute_statistics(
    fake_loader, feature_extractor, device
)

# FID
fid_value = calculate_fid(
    mu_real,
    sigma_real,
    mu_fake,
    sigma_fake
)

print(f"\nFID (MNIST real vs cGAN): {fid_value:.2f}")

