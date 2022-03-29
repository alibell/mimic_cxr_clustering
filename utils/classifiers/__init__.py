from torch import nn, optim
from torchvision.models import mobilenet_v3_small
from xgboost import train
from ..mblocks import M_Block
import torch
import copy
import numpy as np

#
# All classifiers should inherit from ImageClassifier
#

def sigmoid(x):
    return  1/(1 + np.exp(-x))

class ImageClassifier(nn.Module):
    def __init__ (self, n_labels=9, weight_balance=True, weights=None):
        super().__init__()

        # Getting the weights for weight balancing
        self.weight_balance = weight_balance
        self.weights = weights

        self.loss_fn = nn.BCELoss()
        if weight_balance == True:
            self.loss_fn.weight = torch.ones((2, n_labels))

    def forward (self, x):
        return NotImplementedError

    def predict_proba (self, x):
        self.eval()

        if isinstance(x, list):
            x = x[0]

        with torch.no_grad():
            y_hat = self.forward(x)

            return y_hat.cpu().numpy()

    def predict (self, x):

        if isinstance(x, list):
            x = x[0]

        y_hat = self.predict_proba(x)

        return (y_hat >= 0.5)*1

    def fit (self, x, y):
        
        # Weight balancing
        if self.weight_balance and self.weights is not None:
            self.loss_fn.weight = y*self.weights[1,:] + (1-y)*self.weights[0,:]

        self.train()
        self.optimizer.zero_grad()

        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        loss.backward()

        self.optimizer.step()

        loss_scalar = loss.detach().cpu().item()
        
        return [loss_scalar]

    def save_model (self, path):
        torch.save(self.state_dict(), path)

    def load_model (self, path):
        state_dict = torch.load(path)
        self.loss_fn.weight = torch.ones(state_dict["loss_fn.weight"].shape)
        
        self.load_state_dict(state_dict)

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

# naiveImageClassifier without MobileNet

class naiveImageClassifierVanilla (ImageClassifier):

    def __init__ (self, n_labels=9, weight_balance=True, weights=None):
        super().__init__(n_labels=n_labels, weight_balance=weight_balance, weights=weights)

        # Layer that duplicate the color layer (b&w to rgb)
        duplicate_color_layers = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,1), padding="same")
        duplicate_color_layers.requires_grad_(False)
        duplicate_color_layers.load_state_dict({
            "weight":torch.ones(duplicate_color_layers.weight.shape, dtype=torch.float32),
            "bias":torch.zeros(duplicate_color_layers.bias.shape)
        })

        self.classification_layer = nn.Sequential(
            nn.Conv2d(1, 3, (3,3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(3, 6, (3,3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(6, 12, (3,3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(12, 24, (3,3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(24, 48, (3,3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(48, 96, (3,3), padding="same"),
            nn.BatchNorm2d(96),
            nn.MaxPool2d((2,2)),
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
        y_hat = self.classification_layer(x)

        return y_hat

# Classical AE classifiers

class AEClassifier (ImageClassifier):
    def __init__ (self, pretrained, n_labels=9, weight_balance=True, weights=None, train_ae=True):
        """
            Parameters:
            -----------
            n_labels: int, number of labels
            weight_balance: boolean, if true the weight is balanced in the CE Loss
            weights: dict, weights of the labels
            train_ae: boolean, if true the AE is re-trained
        """
        super().__init__(n_labels=n_labels, weight_balance=weight_balance, weights=weights)

       # Getting backbone
        self.encoder = copy.deepcopy(pretrained.cpu())
        for param in self.encoder.parameters():
            param.requires_grad = train_ae

        self.classification_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, n_labels)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        if weight_balance == True:
            self.loss_fn.weight = torch.ones((2, n_labels))
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward (self, x):
        x_encoded = self.encoder(x)["latent_space"]

        # Classification layer from the last block
        y_hat = self.classification_layer(x_encoded)

        return y_hat

    def predict_proba (self, x):
        y_hat = super().predict_proba(x)
        y_hat = sigmoid(y_hat)

        return y_hat

# Classical from embeddings
class AEClassifier_Embeddings (AEClassifier):
    """
        Classify from the embedding reconstruction of the AE

        Parameters:
        -----------
        n_labels: int, number of labels
        weight_balance: boolean, if true the weight is balanced in the CE Loss
        weights: dict, weights of the labels
        train_ae: boolean, if true the AE is re-trained
    """
    def __init__ (self, pretrained, n_labels=9, weight_balance=True, weights=None, train_ae=True):
        super().__init__(pretrained=pretrained, n_labels=n_labels, weight_balance=weight_balance, weights=weights, train_ae=train_ae)

        self.classification_layer = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, n_labels)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward (self, x):
        x_encoded = self.encoder.fullpass(x, embedding=True)[1]

        # Classification layer from the last block
        y_hat = self.classification_layer(x_encoded)

        return y_hat

# M-Blocks AE classifiers

class MBlockAEClassifier (ImageClassifier):
    """
        Parameters:
        -----------
        n_labels: int, number of labels
        weight_balance: boolean, if true the weight is balanced in the CE Loss
        weights: dict, weights of the labels
        train_ae: boolean, if true the AE is re-trained
    """
    def __init__ (self, pretrained, n_labels=9, weight_balance=True, weights=None, train_ae=True):
        super().__init__(n_labels=n_labels, weight_balance=weight_balance, weights=weights)

       # Getting backbone
        self.encoder = copy.deepcopy(pretrained.cpu())
        for param in self.encoder.parameters():
            param.requires_grad = train_ae

        # Declaring M-Blocks
        self.m_blocks = nn.ModuleDict({
            "0": nn.ModuleList([
                M_Block(1, 64),
                M_Block(64, 128),
                M_Block(128, 256),
                M_Block(256, 512)
            ]),
            "1": nn.ModuleList([
              M_Block(32, 96),
              M_Block(96, 192),
              M_Block(192, 384)
            ]),
            "2": nn.ModuleList([
              M_Block(64, 144),                                
              M_Block(144, 288)
            ]),
            "3": nn.ModuleList([
              M_Block(104, 216),                               
            ])
        })

        self.classification_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, n_labels)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        if weight_balance == True:
            self.loss_fn.weight = torch.ones((2, n_labels))

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward (self, x):
        x_encoded = self.encoder(x)["intermediates"]
        x_encoded = [x] + x_encoded

        # Applying the M-Blocks
        for level in self.m_blocks.keys():
            x_encoded_level = []
            for left, right, i in zip(x_encoded[0:-1], x_encoded[1:], range(len(x_encoded)-1)):
                    x_encoded_level.append(
                        self.m_blocks[str(level)][i](left, right)
                    )
            x_encoded = x_encoded_level

        # Classification layer from the last block
        y_hat = self.classification_layer(x_encoded[0])

        return y_hat

    def predict_proba (self, x):
        y_hat = super().predict_proba(x)
        y_hat = sigmoid(y_hat)

        return y_hat