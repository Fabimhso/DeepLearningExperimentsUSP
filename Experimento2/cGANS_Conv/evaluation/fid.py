import torch
import numpy as np
from scipy import linalg

def extract_features(model, dataloader, device):
    """
    Extrai features intermediárias de uma CNN para cálculo do FID

    Parâmetros:
    - model: CNN treinada (usada como extrator)
    - dataloader: DataLoader (real ou fake)
    - device: cpu ou cuda

    Retorno:
    - features: array numpy (N, D)
    """

    model.eval()               # modo avaliação
    features = []              # lista para armazenar features

    with torch.no_grad():      # desativa gradientes
        for images, _ in dataloader:
            images = images.to(device)

            # Forward apenas até a camada convolucional
            x = model.conv1(images)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)

            x = model.conv2(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)

            # Flatten das features
            x = x.view(x.size(0), -1)

            features.append(x.cpu().numpy())

    # Concatena todos os batches
    features = np.concatenate(features, axis=0)
    return features


def compute_statistics(features):
    """
    Calcula média e covariância das features

    Parâmetros:
    - features: array (N, D)

    Retorno:
    - mu: média (D,)
    - sigma: covariância (D, D)
    """

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Calcula Fréchet Inception Distance (FID)

    Fórmula:
    ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1Σ2)¹ᐟ²)
    """

    diff = mu1 - mu2

    # Produto das covariâncias
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Correção numérica (parte imaginária pequena)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
