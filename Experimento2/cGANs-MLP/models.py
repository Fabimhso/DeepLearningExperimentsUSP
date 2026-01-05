# =========================================================
# models.py
# Contém apenas definições de ARQUITETURA CNN e cGANs
# Para cGANS utiliza uma MLP
# (nenhum treinamento aqui)
# =========================================================

# Importa o PyTorch base
import torch

# Importa o módulo de redes neurais
import torch.nn as nn

# =========================================================
# CNN CLASSIFICADORA PARA MNIST
# =========================================================
class CNNClassifier(nn.Module):
    """
    CNN simples usada para:
    - Classificação com dados reais
    - Classificação com dados fake gerados pela cGAN
    """

    def __init__(self):
        # Inicializa a classe base nn.Module
        super().__init__()

        # -------------------------
        # Extrator de características
        # -------------------------
        self.features = nn.Sequential(

            # Primeira convolução:
            # Entrada: 1 canal (imagem MNIST)
            # Saída: 32 mapas de características
            # kernel 3x3
            # Paramentros: numero de filtros=32, tamanho do kernel = 3
            nn.Conv2d(1, 32, kernel_size=3),

            # Função de ativação ReLU
            nn.ReLU(),

            # Reduz resolução espacial pela metade
            nn.MaxPool2d(2),
            #saida mapa 13x33

            # Segunda convolução:
            # Entrada: 32 mapas
            # Saída: 64 mapas
            nn.Conv2d(32, 64, kernel_size=3),

            # Ativação
            nn.ReLU(),

            # Novo downsampling
            nn.MaxPool2d(2)
        )
        # saida mapa 5x5

        # -------------------------
        # Classificador totalmente conectado
        # -------------------------
        self.classifier = nn.Sequential(

            # Achata o tensor 4D → 2D
            nn.Flatten(),

            # Camada totalmente conectada
            nn.Linear(64 * 5 * 5, 128),

            # Ativação
            nn.ReLU(),

            # Camada de saída (10 classes)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        Define o fluxo forward da CNN
        """

        # Extrai características
        x = self.features(x)

        # Realiza tarefa de Classificação
        x = self.classifier(x)

        return x


# =========================================================
# GERADOR CONDICIONAL (cGAN)
# =========================================================
class ConditionalGenerator(nn.Module):
    """
    Gerador da cGAN:
    Entrada:
      - vetor de ruído z
      - rótulo y
    Saída:
      - imagem 28x28 condicionada ao rótulo
    """

    def __init__(self, z_dim=100, n_classes=10):
        super().__init__()

        # Embedding transforma rótulos em vetores
        self.label_emb = nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=n_classes
        )

        # Rede totalmente conectada (MLP)
        self.net = nn.Sequential(

            # Entrada: ruído + rótulo
            nn.Linear(z_dim + n_classes, 256),

            nn.ReLU(),

            nn.Linear(256, 512),

            nn.ReLU(),

            # Saída: 28*28 pixels
            nn.Linear(512, 784),

            # Tanh → compatível com normalização [-1, 1]
            nn.Tanh()
        )

    def forward(self, z, y):
        """
        Forward do gerador
        """

        # Converte rótulo inteiro em vetor embedding
        y_emb = self.label_emb(y)

        # Concatena ruído e rótulo
        x = torch.cat([z, y_emb], dim=1)

        # Gera imagem e reorganiza para formato 2D
        img = self.net(x).view(-1, 1, 28, 28)

        return img


# =========================================================
# DISCRIMINADOR CONDICIONAL (cGAN)
# =========================================================
class ConditionalDiscriminator(nn.Module):
    """
    Discriminador da cGAN:
    Entrada:
      - imagem
      - rótulo
    Saída:
      - probabilidade de ser real
    """

    def __init__(self, n_classes=10):
        super().__init__()

        # Embedding do rótulo no espaço da imagem
        self.label_emb = nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=784
        )

        # Rede discriminadora - MLP
        self.net = nn.Sequential(

            nn.Linear(784 * 2, 512), #tamanho da imagem x (dados + classe)
                                      #vetor de classe codificado como 784 valores
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),

            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),

            # Saída probabilística
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        Forward do discriminador
        """

        # Achata imagem
        x = x.view(x.size(0), -1)

        # Embedding do rótulo
        y_emb = self.label_emb(y)

        # Concatena imagem + rótulo
        d_input = torch.cat([x, y_emb], dim=1)

        # Classificação real/fake
        out = self.net(d_input)

        return out
