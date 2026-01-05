import torch
import torch.nn as nn
import torch.optim as optim

def train_cgan(G, D, dataloader, device, epochs=30, z_dim=100):
    """
    Treinamento padrão de uma DC-cGAN
    """
    criterion = nn.BCELoss()
    opt_g = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_imgs, labels in dataloader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            batch = real_imgs.size(0)

            # Rótulos reais e falsos
            valid = torch.ones(batch, 1, device=device)
            fake = torch.zeros(batch, 1, device=device)

            # ======================
            # Treina Discriminador
            # ======================
            z = torch.randn(batch, z_dim, device=device)
            gen_imgs = G(z, labels)

            d_real = D(real_imgs, labels)
            d_fake = D(gen_imgs.detach(), labels)

            loss_d = criterion(d_real, valid) + criterion(d_fake, fake)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ======================
            # Treina Gerador
            # ======================
            z = torch.randn(batch, z_dim, device=device)
            gen_imgs = G(z, labels)

            loss_g = criterion(D(gen_imgs, labels), valid)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")
