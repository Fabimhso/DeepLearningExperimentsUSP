import torch
import torch.nn as nn
import torch.optim as optim

def train_cnn(model, dataloader, device, epochs=10):
    """
    Treinamento supervisionado da CNN
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}")
