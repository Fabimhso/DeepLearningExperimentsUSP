# Treina uma CNN
# Extrai caracretisicas
# Treina MLP e SVM
# Treina Autoencoder
# Gera vetor z
# Treina MLP e SVM

# ==========================================================
# IMPORTS
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs_cnn = 10
epochs_ae = 15
latent_dim = 64
num_classes = 10

# ==========================================================
# CNN PARA CLASSIFICAÇÃO E EXTRAÇÃO DE FEATURES
# ==========================================================
class CNNFeatureExtractor(nn.Module):
    """
    CNN simples para MNIST
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        features = x.view(x.size(0), -1)

        if return_features:
            return features

        return self.fc(features)

# ==========================================================
# AUTOENCODER (REDUÇÃO DE DIMENSIONALIDADE)
# ==========================================================
class Autoencoder(nn.Module):
    """
    Autoencoder totalmente conectado
    """
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ==========================================================
# TREINAMENTO DA CNN
# ==========================================================
def train_cnn(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs_cnn):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            loss = criterion(model(x), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[CNN] Epoch {epoch+1}/{epochs_cnn} | Loss: {loss.item():.4f}")

# ==========================================================
# TREINAMENTO DO AUTOENCODER
# ==========================================================
def train_autoencoder(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs_ae):
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)

            loss = criterion(model(x), x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[AE] Epoch {epoch+1}/{epochs_ae} | Loss: {loss.item():.4f}")

# ==========================================================
# EXTRAÇÃO DE FEATURES
# ==========================================================
def extract_cnn_features(model, loader):
    model.eval()
    X, y = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images, return_features=True)

            X.append(feats.cpu().numpy())
            y.append(labels.numpy())

    return np.vstack(X), np.hstack(y)

def extract_ae_features(model, loader):
    model.eval()
    X, y = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            z = model.encoder(images)

            X.append(z.cpu().numpy())
            y.append(labels.numpy())

    return np.vstack(X), np.hstack(y)

# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=range(num_classes)
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.grid(False)
    plt.show()

# ==========================================================
# DATASET MNIST
# ==========================================================
transform = transforms.ToTensor()

train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ==========================================================
# 1) TREINA CNN
# ==========================================================
cnn = CNNFeatureExtractor().to(device)
train_cnn(cnn, train_loader)

# ==========================================================
# 2) TREINA AUTOENCODER
# ==========================================================
autoencoder = Autoencoder().to(device)
train_autoencoder(autoencoder, train_loader)

# ==========================================================
# 3) EXTRAI FEATURES
# ==========================================================
X_train_cnn, y_train = extract_cnn_features(cnn, train_loader)
X_test_cnn,  y_test  = extract_cnn_features(cnn, test_loader)

X_train_ae, _ = extract_ae_features(autoencoder, train_loader)
X_test_ae,  _ = extract_ae_features(autoencoder, test_loader)

# ==========================================================
# 4)  CLASSIFICADORES SOBRE FEATURES CNN
# ==========================================================
mlp_cnn = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300)
mlp_cnn.fit(X_train_cnn, y_train)
y_pred = mlp_cnn.predict(X_test_cnn)
plot_confusion(y_test, y_pred, "MLP – Features CNN")

svm_cnn = SVC(kernel="rbf", gamma="scale")
svm_cnn.fit(X_train_cnn, y_train)
y_pred = svm_cnn.predict(X_test_cnn)
plot_confusion(y_test, y_pred, "SVM – Features CNN")

# ==========================================================
# 5)  CLASSIFICADORES SOBRE FEATURES AUTOENCODER
# ==========================================================
mlp_ae = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)
mlp_ae.fit(X_train_ae, y_train)
y_pred = mlp_ae.predict(X_test_ae)
plot_confusion(y_test, y_pred, "MLP – Features Autoencoder")

svm_ae = SVC(kernel="rbf", gamma="scale")
svm_ae.fit(X_train_ae, y_train)
y_pred = svm_ae.predict(X_test_ae)
plot_confusion(y_test, y_pred, "SVM – Features Autoencoder")
