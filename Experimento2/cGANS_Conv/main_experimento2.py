# Treina uma CNN sobre dados reais
# Testa uma CNN sobre dados reais
# Treina uma cGANs convolucional
# Gera dadps fakes
# Treina uma CNN sobre dados reais + fake
# Testa uma CNN sobre dados reais
# Grafico da matriz de confusão

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# ======================
# Importação dos módulos
# ======================
from models.cnn import CNNClassifier
from models.cgan_generator import CGANGenerator
from models.cgan_discriminator import CGANDiscriminator

from training.train_cnn import train_cnn
from training.train_cgan import train_cgan

from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.fid import extract_features, compute_statistics, calculate_fid

# ======================
# Configurações gerais
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

batch_size = 128 #tamanho do batch
z_dim = 100 #dimensão de z

# ======================
# Transformações do MNIST
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # compatível com Tanh do gerador
])

# ======================
# Carregamento do MNIST
# ======================
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ==========================================================
# CNN treinada com dados reais
# ==========================================================
print("\nTreinando CNN com dados reais (MNIST)...")

cnn_real = CNNClassifier().to(device)
train_cnn(
    model=cnn_real,
    dataloader=train_loader,
    device=device,
    epochs=10
)

# ==========================================================
# Treinamento da cGAN convolucional
# ==========================================================
print("\nTreinando cGAN convolucional...")

G = CGANGenerator(z_dim=z_dim).to(device)
D = CGANDiscriminator().to(device)

train_cgan(
    G=G,
    D=D,
    dataloader=train_loader,
    device=device,
    epochs=3,
    z_dim=z_dim
)

# ==========================================================
# Geração do dataset fake condicional
# ==========================================================
print("\nGerando dataset fake com a cGAN...")

def generate_fake_dataset(generator, n_samples=60000):
    """
    Gera um dataset sintético rotulado usando a cGAN
    """
    generator.eval()

    z = torch.randn(n_samples, z_dim, device=device)
    labels = torch.randint(0, 10, (n_samples,), device=device)

    with torch.no_grad():
        images = generator(z, labels)

    return TensorDataset(images.cpu(), labels.cpu())

fake_dataset = generate_fake_dataset(G)
fake_loader = DataLoader(
    fake_dataset,
    batch_size=batch_size,
    shuffle=True
)

# ==========================================================
# CNN treinada apenas com dados fake
# ==========================================================
print("\nTreinando CNN com dados fake (cGAN)...")

cnn_fake = CNNClassifier().to(device)
train_cnn(
    model=cnn_fake,
    dataloader=fake_loader,
    device=device,
    epochs=10
)

# ==========================================================
# Matrizes de confusão (avaliação no teste real)
# ==========================================================
print("\nMatriz de confusão – CNN treinada com dados reais")
plot_confusion_matrix(
    model=cnn_real,
    dataloader=test_loader,
    device=device,
    title="CNN treinada com MNIST real"
)

print("\nMatriz de confusão – CNN treinada com dados fake (cGAN)")
plot_confusion_matrix(
    model=cnn_fake,
    dataloader=test_loader,
    device=device,
    title="CNN treinada com MNIST gerado por cGAN"
)

# ==========================================================
# FID – Qualidade das imagens geradas
# ==========================================================
print("\nCalculando FID (MNIST real vs cGAN)...")

# Extrai features dos dados reais
real_features = extract_features(
    model=cnn_real,
    dataloader=train_loader,
    device=device
)

# Extrai features dos dados fake
fake_features = extract_features(
    model=cnn_real,      # mesma CNN como extrator
    dataloader=fake_loader,
    device=device
)

# Estatísticas
mu_real, sigma_real = compute_statistics(real_features)
mu_fake, sigma_fake = compute_statistics(fake_features)

# FID
fid_value = calculate_fid(
    mu_real,
    sigma_real,
    mu_fake,
    sigma_fake
)

print(f"\nFID (MNIST real vs cGAN): {fid_value:.2f}")
