import torch
import copy
from torch import nn, optim
from functools import reduce
from operator import add
from torchvision import transforms

class UNetConv (nn.Module):
    """
        UNet convolution block
    """
    def __init__ (self, in_channels, out_channels, kernel_size, padding=0, batchnorm = True):
        """
            Perform a basic UNET Convolution +/- BatchNorm operation
            
            Parameters:
            -----------
            in_channels: numbers of input layers
            out_channels: numbers of output layers
            kernel_size: convolution kernel size
            padding: convolution padding parameter
            batchnorm: boolean, if true a batchnorm layer is added

            Output:
            ------
            Tensor after applying the convolutions (n, out_channels, height, width)
        """
        super().__init__()
        
        unet_blocks = []

        # Adding conv block
        for i, in_channel, out_channel in zip(range(2), [in_channels, out_channels], [out_channels, out_channels]):
            unet_blocks += [
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding),
                nn.ReLU()
            ]
            if batchnorm:
                unet_blocks.append(
                    nn.BatchNorm2d(out_channels)
                )

        self.unet_block = nn.Sequential(*unet_blocks)

    def forward(self, x):
        return self.unet_block(x)

class UNetDownConv (UNetConv):
    """
        UNet block for encoder
        This block is used to apply a Unet block + a convolution block

        Parameters:
        -----------
        self, in_channels, out_channels, kernel_size, padding, batchnorm: same as UNetConv
        down_factor: int, the factor of downsampling

        Output:
        -------
        Tuple (x1, x2)
        x1 is the tensor before downsampling of size (n, out_channels, height, width)
        x1 is the tensor after downsampling of size (n, out_channels, height/down_factor, width/down_factor)
    """
    def __init__ (self, in_channels, out_channels, kernel_size, padding=0, batchnorm=True, down_factor=2):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, batchnorm = batchnorm)

        self.downblock = nn.MaxPool2d(kernel_size=down_factor)

    def forward (self, x):
        x1 = self.unet_block(x)
        x2 = self.downblock(x1)

        return x1, x2

class UNetUpConv (UNetConv):
    """
        UNet block for decoder
        This block is used to apply a Unet block + a upsampling block
        Two types of upconvolution :
            transposeConv or Upsampling

        Parameters:
        -----------
        self, in_channels, out_channels, kernel_size, padding, batchnorm: same as UNetConv
        up_mode: str, if transposeConv a transpose convolution is performed, an Upsampling+Convolution is performed otherwise
        up_factor: int, the factor of upsampling

        Output:
        -------
        Tuple (x1, x2)
        x1 is the tensor before upsampling of size (n, out_channels, height, width)
        x1 is the tensor after upsampling of size (n, out_channels, height*up_factor, width*up_factor)
    """
    def __init__ (self, in_channels, out_channels, kernel_size, up_mode="upSample", padding=0, batchnorm=True, up_factor=2):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, batchnorm = batchnorm)

        # Adding Up
        if up_mode == "transposeConv":
            self.upblock = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=up_factor)
        else:
            self.upblock = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=up_factor, align_corners=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=padding)
            )

    def forward (self, x):
        x1 = self.unet_block(x)
        x2 = self.upblock(x1)

        return x1, x2

class skipConnection (nn.Module):
    """
        Skip connecter layer
        This layer get two images, the left one (x) and right one (y).
        It center crop the left image and concatenate them.

        Parameters:
        -----------
        in_channels: numbers of input layers
        out_channels: numbers of output layers
    """
    def __init__ (self, in_channels, out_channels):
        super().__init__()
        self.merge_convolution = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding="same")

    def forward(self, x, y):
        cropper = transforms.CenterCrop(y.shape[2:])
        x_cropped = cropper(x)

        xy = torch.cat([x_cropped, y], axis=1)
        xy = self.merge_convolution(xy)

        return xy

class cxr_unet_ae (nn.Module):
    """
        CXR UNET AE Class model
    """

    def __init__ (self, noise_variance=1e-2, p_salt_pepper_noise=0.2):
        """
        Parameters:
        -----------
        noise_variance: float, variance of the gaussian noise for this denoising auto-encoder
        p_salt_pepper_noise: float, proportion of salt and pepper noise
        """
        super().__init__()

        # Getting backbone    
        self.noise_variance = noise_variance**(0.5)
        self.p_salt_pepper_noise = p_salt_pepper_noise

        # Encoder
        self.encoder_params = [
            {"in_channels":1, "out_channels":64, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "down_factor":2},
            {"in_channels":64, "out_channels":128, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "down_factor":2},
            {"in_channels":128, "out_channels":256, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "down_factor":2},
            {"in_channels":256, "out_channels":512, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "down_factor":2},
        ]
        self.encoder_modules = nn.ModuleList(
            [UNetDownConv(**encoder_params) for encoder_params in self.encoder_params]
        )
        self.encoder_final_layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1,1), padding="same")
        )

        # Decoder
        self.decoder_params = [
            {"in_channels":1024, "out_channels":512, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "up_factor":2},
            {"in_channels":512, "out_channels":256, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "up_factor":2},
            {"in_channels":256, "out_channels":128, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "up_factor":2},
            {"in_channels":128, "out_channels":64, "kernel_size":(3,3), "padding":"same", "batchnorm":True, "up_factor":2}
        ]
        encoders_params = copy.copy(self.encoder_params)
        encoders_params.reverse() # Reversing the list
        encoders_params = [None] + encoders_params # To not process skip connection at first step
        self.decoder_modules = nn.ModuleList(
            [
                nn.ModuleList([
                    skipConnection(in_channels=(decoder_params["in_channels"]+encoders_param["out_channels"]) if encoders_param is not None else decoder_params["in_channels"], out_channels=decoder_params["in_channels"]),
                    UNetUpConv(**decoder_params)
                ]) for decoder_params, encoders_param  in zip(self.decoder_params, encoders_params)
            ]
        )
        self.decoder_final_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), padding="same")
        )


        self.reconstruction_loss = nn.MSELoss()

    def generate_noisy_image (self, images, variance, p_salt_pepper):
        if variance != 0:
            variance_noise = (variance**0.5)*torch.randn_like(images).to(images.device)
            images = images + variance_noise
        
        if p_salt_pepper != 0:
            salt_pepper_noise = ((torch.rand(images.shape) >= p_salt_pepper)*1).to(images.device)
            images = images*salt_pepper_noise

        return images

    def encoder (self, x):
        encoder_intermediates = []
        for encoder_module in self.encoder_modules:
            x_before, x = encoder_module(x)
            encoder_intermediates.append(x_before)
        
        x = self.encoder_final_layer(x)

        return encoder_intermediates, x

    def decoder (self, encoder_intermediates, x):
        encoder_intermediates.reverse() # Reversing the list
        encoder_intermediates = [None] + encoder_intermediates # To not process skip connection at first step
        for decoder_model, intermediate in zip(self.decoder_modules, encoder_intermediates):
            if intermediate is not None:
                x = decoder_model[0](intermediate, x)
            x = decoder_model[1](x)[1]
        output = self.decoder_final_layer(x)

        return output

    def compute_loss (self, y_hat, x, y):
        raise NotImplementedError()


    def forward (self, x):
        encoder_intermediates, x = self.encoder(x)

        out = {
            "intermediates":encoder_intermediates,
            "latent_space":x
        }

        return out

    def fullpass (self, x):
        raise NotImplementedError()

    def fit (self, x, y):
        self.train()
        self.optimizer.zero_grad()

        # Creatining x with random noise
        x_with_noise = self.generate_noisy_image(x, variance=self.noise_variance, p_salt_pepper=self.p_salt_pepper_noise)

        y_hat = self.fullpass(x_with_noise)
        loss = self.compute_loss(y_hat, x, y)
        loss_sum = reduce(add, loss)

        loss_sum.backward()
        self.optimizer.step()

        loss = [x.detach().cpu().item() for x in loss]

        return loss

    def predict (self, x, mode="fullpass"):
        self.eval()

        with torch.no_grad():
            if mode == "encoder":
                y_hat = self.encoder(x)
            else:
                y_hat = self.fullpass(x)

        return y_hat

    def save_model (self, path):
        torch.save(self.state_dict(), path)

    def load_model (self, path):
        self.load_state_dict(torch.load(path))

        return self

class cxr_unet_ae_1 (cxr_unet_ae):
    """
        CXR AE 1 - Conv-Deconv Unet auto-encoder without report embedding loss
    """

    def __init__ (self, noise_variance=1e-2, p_salt_pepper_noise=0.2):
        super().__init__(noise_variance=noise_variance, p_salt_pepper_noise=p_salt_pepper_noise)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def compute_loss(self, y_hat, x, y=None):

        # Center crop x to compare it with y_hat
        cropper = transforms.CenterCrop(y_hat.shape[2:])
        x_ = cropper(x)

        loss_reconstruction = self.reconstruction_loss(y_hat, x_)

        loss = [loss_reconstruction]

        return loss

    def fullpass (self, x):

        encoder_intermediates, y_encode = self.encoder(x)
        y_decode = self.decoder(encoder_intermediates, y_encode)

        return y_decode

class cxr_unet_ae_2 (cxr_unet_ae):
    """
        CXR AE 1 - Conv-Deconv Unet auto-encoder without report embedding loss
    """

    def __init__ (self, embedding_size=768, noise_variance=1e-2, p_salt_pepper_noise=0.2):
        super().__init__(noise_variance=noise_variance, p_salt_pepper_noise=p_salt_pepper_noise)

        self.embedding_size = embedding_size
        self.embeddings_decoder = nn.Sequential(
            nn.Conv2d(1024, 512, (3,3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(512, 512, (3,3), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(512, 512, (1,1), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size)
        )
        self.embedding_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)

    def compute_loss(self, y_hat, x, y=None):

        # Center crop x to compare it with y_hat
        y_hat_reconstruction, y_hat_embedding = y_hat
        y_embedding = y

        cropper = transforms.CenterCrop(y_hat_reconstruction.shape[2:])
        x_ = cropper(x)
        loss_reconstruction = self.reconstruction_loss(y_hat_reconstruction, x_)
        loss_embedding = self.embedding_loss(y_hat_embedding, y_embedding)

        loss = [loss_reconstruction, loss_embedding]

        return loss

    def predict (self, x, mode="fullpass", embedding=False):
        self.eval()

        with torch.no_grad():
            if mode == "encoder":
                y_hat = self.encoder(x)
            else:
                y_hat = self.fullpass(x, embedding=embedding)

        return y_hat        

    def fullpass (self, x, embedding=True):

        encoder_intermediates, y_encode = self.encoder(x)
        y_decode = self.decoder(encoder_intermediates, y_encode)
        if embedding:
            y_embedding = self.embeddings_decoder(y_encode)
        else:
            y_embedding = None

        return y_decode, y_embedding