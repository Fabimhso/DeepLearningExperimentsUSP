import torch
import torch.nn as nn
import torch.optim as optim

def train_caae(E, D, C, dataloader, device, epochs=10):
    """
    Treinamento do Autoencoder Adversarial Condicional
    E: Encoder
    D: Decoder
    C: Discriminador latente
    """

    recon_loss = nn.MSELoss()
    adv_loss = nn.BCELoss()

    opt_E = optim.Adam(E.parameters(), lr=2e-4)
    opt_D = optim.Adam(D.parameters(), lr=2e-4)
    opt_C = optim.Adam(C.parameters(), lr=2e-4)

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch = imgs.size(0)

            valid = torch.ones(batch, 1, device=device)
            fake = torch.zeros(batch, 1, device=device)

            # ======================
            # Discriminador latente
            # ======================
            z_real = torch.randn(batch, E.fc.out_features, device=device)
            z_fake = E(imgs)

            loss_C = adv_loss(C(z_real), valid) + adv_loss(C(z_fake.detach()), fake)

            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

            # ======================
            # Encoder + Decoder
            # ======================
            z = E(imgs)
            recon = D(z, labels)

            loss_recon = recon_loss(recon, imgs)
            loss_adv = adv_loss(C(z), valid)

            loss_ED = loss_recon + 0.01 * loss_adv

            opt_E.zero_grad()
            opt_D.zero_grad()
            loss_ED.backward()
            opt_E.step()
            opt_D.step()

        print(f"Epoch {epoch+1}/{epochs} | Recon: {loss_recon.item():.4f} | Adv: {loss_adv.item():.4f}")
