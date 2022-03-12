from functools import partial
from torchvision.transforms import RandomResizedCrop, RandomCrop, Resize, Normalize, RandomRotation
import torch
import numpy as np

TORCH_IMAGE_MEAN = [0.485, 0.456, 0.406]
TORCH_IMAGE_STD = [0.229, 0.224, 0.225]

def numpy_to_tensor (x):
    return torch.tensor(x).unsqueeze(0)

def tensor_to_numpy (x):
    x = x.detach().cpu()

    x = x.reshape(x.shape[-2], x.shape[-1])

    return x.numpy()

def preprocess_image (x, size=None):

    mean = np.array(TORCH_IMAGE_MEAN).mean()
    std = np.array(TORCH_IMAGE_STD).sum()/np.sqrt(3)
    normalizer = Normalize(mean=mean, std=std)
    if size is not None:
        resizer = Resize(size)
    
    output = normalizer(x/255.)
    if size is not None:
        output = resizer(output)

    return output

#
# Collate functions :
# Functions that collate togethers images of different size for the dataloader
#

def collate_fn (x, size=500, rotate=True, resize_crop=True, crop=True):
    """
        collate_fn

        Function that collate togethers images of different size for the dataloader
        It performs normalizing, resizing, cropping and stacking images inside  a tensor

        Parameters:
        -----------
        size: int, resized image size
        rotate: boolean, if true a random rotate is applied
        resize_crop: boolean, if true a random resized crop is applied otherwise it is a just a simple random crop
        crop: boolean, if False no crop is applied at all
    """

    resizer = Resize(size)
    if resize_crop:
        random_croper = RandomResizedCrop(size=(size,size))
    else:
        random_croper = RandomCrop(size=(size, size))
    random_rotater = RandomRotation(20)

    # Applying randomCrop
    images_tensor = []

    for x_ in x:
        image_tensor = torch.tensor(x_, dtype=torch.float32).unsqueeze(dim=0)
        image_tensor = resizer(image_tensor)
        if crop:
            image_tensor = random_croper(image_tensor)
        if rotate:
            image_tensor = random_rotater(image_tensor)

        images_tensor.append(image_tensor)

    output = torch.stack(images_tensor)
    output = preprocess_image(output, size=None)

    return output

def get_collater(size=500, rotate=True, resize_crop=True, crop=True):
    """
        Provide a configured collater function

        Parameters:
        -----------
        size: int, resized image size
        rotate: boolean, if true a random rotate is applied
        resize_crop: boolean, if true a random resized crop is applied otherwise it is a just a simple random crop
        crop: boolean, if true 
    """

    return partial(collate_fn, size=size, rotate=rotate, resize_crop=resize_crop, crop=crop)