from torch import nn, optim
from torchvision.models import mobilenet_v3_small
import torch

#
# All classifiers should inherit from ImageClassifier
#

class ImageClassifier(nn.Module):
    def __init__ (self, n_labels=9, weight_balance=True, weights=None):
        super().__init__()

        # Getting the weights for weight balancing
        self.weight_balance = weight_balance
        self.weights = weights

        self.criterion = nn.BCELoss()

    def forward (self, x):
        return NotImplementedError

    def predict_proba (self, x):
        self.eval()

        with torch.no_grad():
            y_hat = self.forward(x)

            return y_hat.cpu().numpy()

    def predict (self, x):
        y_hat = self.predict_proba(x)

        return (y_hat >= 0.5)*1

    def fit (self, x, y):
        
        # Weight balancing
        if self.weight_balance and self.weights is not None:
            self.criterion.weight = y*self.weights[1,:] + (1-y)*self.weights[0,:]

        self.train()
        self.optimizer.zero_grad()

        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        loss.backward()

        self.optimizer.step()

        loss_scalar = loss.detach().cpu().item()
        
        return [loss_scalar]

# naiveImageClassifier with MobileNet

class naiveImageClassifierMobileNet (ImageClassifier):

    def __init__ (self, n_labels=9, weight_balance=True, weights=None):
        super().__init__(n_labels=n_labels, weight_balance=weight_balance, weights=weights)

       # Getting backbone
        self.mobilenet = mobilenet_v3_small(pretrained=True)

        # Layer that duplicate the color layer (b&w to rgb)
        duplicate_color_layers = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,1), padding="same")
        duplicate_color_layers.requires_grad_(False)
        duplicate_color_layers.load_state_dict({
            "weight":torch.ones(duplicate_color_layers.weight.shape, dtype=torch.float32),
            "bias":torch.zeros(duplicate_color_layers.bias.shape)
        })

        self.encoder = nn.Sequential(*[
            duplicate_color_layers,
            nn.Sequential(*list(self.mobilenet.features.children()))[0:-3],
            nn.Conv2d(96, 48, (3,3), padding="same"),
            nn.BatchNorm2d(48),
        ])

        self.classification_layer = nn.Sequential(
            nn.Conv2d(48, 96, (3,3), padding="same"),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 256, (3,3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, n_labels),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward (self, x):
        y_hat = self.encoder(x)
        y_hat = self.classification_layer(y_hat)

        return y_hat

# Naive image classifier 

class naiveImageClassifierMobileNet (ImageClassifier):

    def __init__ (self, n_labels=9, weight_balance=True, weights=None):
        super().__init__(n_labels=n_labels, weight_balance=weight_balance, weights=weights)

       # Getting backbone
        self.mobilenet = mobilenet_v3_small(pretrained=True)

        # Layer that duplicate the color layer (b&w to rgb)
        duplicate_color_layers = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,1), padding="same")
        duplicate_color_layers.requires_grad_(False)
        duplicate_color_layers.load_state_dict({
            "weight":torch.ones(duplicate_color_layers.weight.shape, dtype=torch.float32),
            "bias":torch.zeros(duplicate_color_layers.bias.shape)
        })

        self.encoder = nn.Sequential(*[
            duplicate_color_layers,
            nn.Sequential(*list(self.mobilenet.features.children()))[0:-3],
            nn.Conv2d(96, 48, (3,3), padding="same"),
            nn.BatchNorm2d(48),
        ])

        self.classification_layer = nn.Sequential(
            nn.Conv2d(48, 96, (3,3), padding="same"),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 256, (3,3), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, n_labels),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward (self, x):
        y_hat = self.encoder(x)
        y_hat = self.classification_layer(y_hat)

        return y_hat
