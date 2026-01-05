import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from models.cnn import CNNClassifier
from models.caae_encoder import CAAEEncoder
from models.caae_decoder import CAAEDecoder
from models.caae_discriminator import CAAEDiscriminator

from training.train_cnn import train_cnn
from training.train_caae import train_caae

from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.fid import extract_features, compute_statistics, calculate_fid

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)

# ======================
# CNN com dados reais
# ======================
cnn_real = CNNClassifier().to(device)
train_cnn(cnn_real, train_loader, device)

# ======================
# Treina cAAE
# ======================
E = CAAEEncoder().to(device)
D = CAAEDecoder().to(device)
C = CAAEDiscriminator().to(device)

train_caae(E, D, C, train_loader, device)

# ======================
# Gera dataset fake
# ======================
def generate_fake(E, D, n=60000):
    z = torch.randn(n, 100, device=device)
    labels = torch.randint(0, 10, (n,), device=device)
    with torch.no_grad():
        images = D(z, labels)
    return TensorDataset(images.cpu(), labels.cpu())

fake_dataset = generate_fake(E, D)
fake_loader = DataLoader(fake_dataset, batch_size=128, shuffle=True)

# ======================
# CNN com dados fake
# ======================
cnn_fake = CNNClassifier().to(device)
train_cnn(cnn_fake, fake_loader, device)

# ======================
# Avaliação
# ======================
plot_confusion_matrix(cnn_real, test_loader, device, "CNN – dados reais")
plot_confusion_matrix(cnn_fake, test_loader, device, "CNN – dados cAAE")

# ======================
# FID
# ======================
real_feat = extract_features(cnn_real, train_loader, device)
fake_feat = extract_features(cnn_real, fake_loader, device)

mu_r, sig_r = compute_statistics(real_feat)
mu_f, sig_f = compute_statistics(fake_feat)

fid = calculate_fid(mu_r, sig_r, mu_f, sig_f)
print(f"FID (MNIST vs cAAE): {fid:.2f}")
