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
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from minisom import MiniSom

# ==========================================================
# CONFIGURAÇÕES GERAIS
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs_cnn = 10

num_classes = 10
som_grid = (10, 10)      # tamanho do mapa SOM
min_cluster_size = 50   # tamanho mínimo para treinar classificador local

# ==========================================================
# CNN PARA CLASSIFICAÇÃO E EXTRAÇÃO DE FEATURES
# ==========================================================
class CNNFeatureExtractor(nn.Module):
    """
    CNN simples para MNIST.
    Também usada como extrator de features profundas.
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
# TREINAMENTO DA CNN
# ==========================================================
def train_cnn(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs_cnn):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[CNN] Epoch {epoch+1}/{epochs_cnn} | Loss: {loss.item():.4f}")

# ==========================================================
# EXTRAÇÃO DE FEATURES DA CNN
# ==========================================================
def extract_features(model, loader):
    model.eval()

    X, y = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images, return_features=True)

            X.append(feats.cpu().numpy())
            y.append(labels.numpy())

    return np.vstack(X), np.hstack(y)

# ==========================================================
# MATRIZ DE CONFUSÃO (FUNÇÃO GENÉRICA)
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
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ==========================================================
# 1) TREINAMENTO DA CNN
# ==========================================================
cnn = CNNFeatureExtractor().to(device)
train_cnn(cnn, train_loader)

# ==========================================================
# AVALIA CNN (BASELINE)
# ==========================================================
cnn.eval()
y_pred_cnn = []

with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        logits = cnn(x)
        y_pred_cnn.extend(torch.argmax(logits, dim=1).cpu().numpy())

y_test = test_data.targets.numpy()
plot_confusion(y_test, y_pred_cnn, "CNN – Dados Reais")

# ==========================================================
# 2) EXTRAÇÃO DE FEATURES
# ==========================================================
X_train, y_train = extract_features(cnn, train_loader)
X_test,  y_test  = extract_features(cnn, test_loader)

print("Features extraídas:", X_train.shape)

# ==========================================================
# 3) BAGGING + MLP
# ==========================================================
#  Parâmetros: numero de estimadores, numero de nuerônios na MLP
bagging_mlp = BaggingClassifier(
    estimator=MLPClassifier(hidden_layer_sizes=(256,), max_iter=300),
    n_estimators=10,
    n_jobs=-1
)

bagging_mlp.fit(X_train, y_train)
y_pred = bagging_mlp.predict(X_test)

plot_confusion(y_test, y_pred, "Bagging + MLP (Features CNN)")

# ==========================================================
# 4) BOOSTING + MLP
# ==========================================================
boosting_mlp = AdaBoostClassifier(
    estimator=MLPClassifier(hidden_layer_sizes=(128,), max_iter=200),
    n_estimators=10
)

boosting_mlp.fit(X_train, y_train)
y_pred = boosting_mlp.predict(X_test)

plot_confusion(y_test, y_pred, "Boosting + MLP (Features CNN)")

# ==========================================================
# 5) SVM GLOBAL
# ==========================================================
svm_global = SVC(kernel="rbf", gamma="scale")
svm_global.fit(X_train, y_train)
y_pred = svm_global.predict(X_test)

plot_confusion(y_test, y_pred, "SVM Global (Features CNN)")

# ==========================================================
# 6) SOM (MAPA AUTO-ORGANIZÁVEL)
# ==========================================================
som = MiniSom(
    som_grid[0], som_grid[1],
    X_train.shape[1],
    sigma=1.0,
    learning_rate=0.5
)

som.random_weights_init(X_train)
som.train_random(X_train, 5000)

# ==========================================================
# ASSOCIAÇÃO AMOSTRA → CLUSTER SOM
# ==========================================================
def som_clusters(som, X):
    return np.array([som.winner(x) for x in X])

train_clusters = som_clusters(som, X_train)
test_clusters  = som_clusters(som, X_test)

# ==========================================================
# 7) SOM + MLP (UM CLASSIFICADOR POR CLUSTER)
# ==========================================================
cluster_mlps = {}
unique_clusters = np.unique(train_clusters, axis=0)

for cluster in unique_clusters:
    idx = np.all(train_clusters == cluster, axis=1)

    if np.sum(idx) < min_cluster_size:
        continue

    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)
    mlp.fit(X_train[idx], y_train[idx])

    cluster_mlps[tuple(cluster)] = mlp

# ==========================================================
# PREDIÇÃO COM SOM + CLASSIFICADORES LOCAIS
# ==========================================================
def predict_som_models(X, clusters, models, fallback):
    preds = []

    for x, c in zip(X, clusters):
        model = models.get(tuple(c))
        if model:
            preds.append(model.predict([x])[0])
        else:
            preds.append(fallback.predict([x])[0])

    return np.array(preds)

y_pred = predict_som_models(
    X_test,
    test_clusters,
    cluster_mlps,
    bagging_mlp    # fallback global
)

plot_confusion(y_test, y_pred, "SOM + MLP (Especialistas Locais)")

# ==========================================================
# 8) SOM + SVM (UM CLASSIFICADOR POR CLUSTER)
# ==========================================================
cluster_svms = {}

for cluster in unique_clusters:
    idx = np.all(train_clusters == cluster, axis=1)

    if np.sum(idx) < min_cluster_size:
        continue

    svm = SVC(kernel="rbf", gamma="scale")
    svm.fit(X_train[idx], y_train[idx])

    cluster_svms[tuple(cluster)] = svm

y_pred = predict_som_models(
    X_test,
    test_clusters,
    cluster_svms,
    svm_global    # fallback global
)

plot_confusion(y_test, y_pred, "SOM + SVM (Especialistas Locais)")