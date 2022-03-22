import torch
from torch import nn, optim
from torchvision.models import mobilenet_v3_small
from functools import reduce
from operator import add

class cxr_ae (nn.Module):
    """
        CXR AE Class model
    """

    def __init__ (self, noise_variance=1e-2):
        super().__init__()

        # Getting backbone
        self.mobilenet = mobilenet_v3_small(pretrained=True)

        # Layer that duplicate the color layer (b&w to rgb)
        duplicate_color_layers = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,1), padding="same")
        duplicate_color_layers.requires_grad_(False)
        duplicate_color_layers.load_state_dict({
            "weight":torch.ones(duplicate_color_layers.weight.shape, dtype=torch.float32),
            "bias":torch.zeros(duplicate_color_layers.bias.shape)
        })
        self.noise_std = noise_variance**(0.5)

        self.encoder = nn.Sequential(*[
            duplicate_color_layers,
            nn.Sequential(*list(self.mobilenet.features.children()))[0:-3],
            nn.Conv2d(96, 48, (3,3), padding="same"),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 24, (3,3), padding="same"),
        ])

        self.decoder = nn.Sequential(*[
            nn.ConvTranspose2d(24, 12, kernel_size=(3,3), stride=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, kernel_size=(3,3), stride=3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, kernel_size=(3,3), stride=2),
            nn.AdaptiveAvgPool2d((500,500))
        ])

        self.reconstruction_loss = nn.MSELoss()

    def compute_loss (self, y_hat, x, y):
        raise NotImplementedError()

    def forward (self, x):
        y_encode = self.encoder(x)

        return y_encode

    def fullpass (self, x):
        raise NotImplementedError()

    def fit (self, x, y):
        self.train()
        self.optimizer.zero_grad()

        # Creatining x with random noise
        noise = self.noise_std*torch.randn_like(x).to(x.device)
        with torch.no_grad():
            x_with_noise = x + noise

        y_hat = self.fullpass(x_with_noise)
        loss = self.compute_loss(y_hat, x, y)
        loss_sum = reduce(add, loss)

        loss_sum.backward()
        self.optimizer.step()

        loss = [x.detach().cpu().item() for x in loss]

        return loss

    def save_model (self, path):
        torch.save(self.state_dict(), path)

    def load_model (self, path):
        self.load_state_dict(torch.load(path))

        return self

class cxr_ae_1 (cxr_ae):
    """
        CXR AE 2 - Conv-Deconv auto-encoder without report embedding loss
    """

    def __init__ (self, noise_variance=1e-2):
        super().__init__(noise_variance=noise_variance)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def compute_loss(self, y_hat, x, y=None):

        loss_reconstruction = self.reconstruction_loss(y_hat, x)

        loss = [loss_reconstruction]

        return loss

    def fullpass (self, x):

        y_encode = self(x)
        y_decode = self.decoder(y_encode)

        return y_decode


class cxr_ae_2 (cxr_ae):
    """
        CXR AE 2 - Conv-Deconv auto-encoder with report embedding loss
    """

    def __init__ (self, embedding_size=768, noise_variance=1e-2):
        super().__init__(noise_variance=noise_variance)

        self.embedding_size = embedding_size

        self.embeddings_decoder = nn.Sequential(
            nn.Conv2d(24, 48, (3,3), padding="same"),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(48, 96, (3,3), padding="same"),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 192, (1,1), padding="same"),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(768, 512)
        )

        self.embeddings_impressions = nn.Sequential(
            nn.Linear(512, self.embedding_size)
        )

        self.embedding_loss = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def compute_loss(self, y_hat, x, y):

        y_hat_reconstruction, y_hat_embedding_impressions = y_hat
        y_embedding_impressions = y

        loss_reconstruction = self.reconstruction_loss(y_hat_reconstruction, x)
        loss_embedding_impressions = self.embedding_loss(y_hat_embedding_impressions, y_embedding_impressions)

        loss = [loss_reconstruction, loss_embedding_impressions]

        return loss

    def fullpass (self, x):

        y_encode = self(x)
        y_decode = self.decoder(y_encode)

        y_embedding_decode = self.embeddings_decoder(y_encode)
        y_embedding_impressions = self.embeddings_impressions(y_embedding_decode)

        return y_decode, y_embedding_impressions
