# =========================================================
# training.py
# Contém rotinas de treinamento e métricas
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import linalg
import matplotlib.pyplot as plt


# =========================================================
# TREINAMENTO DA CNN
# =========================================================
def train_classifier(model, loader, device, epochs=5):
    """
    Treina uma CNN supervisionada
    """

    # Coloca o modelo em modo treino
    model.train()

    # Otimizador Adam
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Função de perda multiclasse
    criterion = nn.CrossEntropyLoss()

    # Loop de épocas
    for epoch in range(epochs):

        # Loop sobre batches
        for x, y in loader:

            # Move dados para GPU/CPU
            x, y = x.to(device), y.to(device)

            # Zera gradientes acumulados
            optimizer.zero_grad()

            # Forward pass
            logits = model(x)

            # Calcula perda
            loss = criterion(logits, y)

            # Backpropagation
            loss.backward()

            # Atualiza pesos
            optimizer.step()

        print(f"[CNN] Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")


# =========================================================
# MATRIZ DE CONFUSÃO
# =========================================================
def plot_confusion(model, loader, device, title):
    """
    Calcula e plota a matriz de confusão
    """

    # Modo avaliação
    model.eval()

    y_true, y_pred = [], []

    # Sem gradientes
    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)

            # Predição
            preds = model(x).argmax(dim=1).cpu()

            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # Visualização
    ConfusionMatrixDisplay(cm).plot()
    plt.title(title)
    plt.show()


# =========================================================
# TREINAMENTO DA cGAN
# =========================================================
def train_cgan(G, D, loader, device, epochs=20, z_dim=100):
    """
    Treina uma cGAN (gerador + discriminador)
    """

    # Otimizadores
    opt_g = optim.Adam(G.parameters(), lr=2e-4)
    opt_d = optim.Adam(D.parameters(), lr=2e-4)

    # Perda adversarial
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for x, y in loader:

            x, y = x.to(device), y.to(device)
            batch = x.size(0)

            # ======================
            # Treina Discriminador
            # ======================
            z = torch.randn(batch, z_dim, device=device)

            fake = G(z, y)

            loss_d = (
                criterion(D(x, y), torch.ones(batch, 1, device=device)) +
                criterion(D(fake.detach(), y),
                          torch.zeros(batch, 1, device=device))
            )

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ======================
            # Treina Gerador
            # ======================
            z = torch.randn(batch, z_dim, device=device)

            fake = G(z, y)

            loss_g = criterion(
                D(fake, y),
                torch.ones(batch, 1, device=device)
            )

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        print(f"[cGAN] Epoch {epoch+1}/{epochs} | "
              f"D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")


# =========================================================
# FID
# =========================================================
def compute_statistics(loader, extractor, device):
    """
    Calcula média e covariância das features
    """

    features = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            features.append(extractor(x).cpu().numpy())

    features = np.concatenate(features)

    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Calcula Fréchet Inception Distance
    """
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)



    # Correção numérica (parte imaginária pequena)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid