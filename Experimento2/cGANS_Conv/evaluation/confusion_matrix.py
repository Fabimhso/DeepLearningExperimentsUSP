import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model, dataloader, device, title):
    """
    Calcula e plota a matriz de confusão de um classificador

    Parâmetros:
    - model: CNN treinada
    - dataloader: loader do conjunto de teste
    - device: cpu ou cuda
    - title: título do gráfico
    """

    model.eval()  # coloca o modelo em modo de avaliação

    all_preds = []   # lista para armazenar predições
    all_labels = []  # lista para armazenar rótulos reais

    # Desativa cálculo de gradiente (economia de memória)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Classe predita = argmax dos logits
            preds = torch.argmax(outputs, dim=1)

            # Armazena resultados
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatena todos os batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calcula a matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)

    # Cria objeto de visualização
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plota
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)

    plt.show()
