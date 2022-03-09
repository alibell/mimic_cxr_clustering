from functools import partial
from torchvision.transforms import RandomResizedCrop, Resize, Normalize
import torch

#
# Collate functions :
# Functions that collate togethers images of different size for the dataloader
#


def collate_fn (x, size=500, ):
    """
        collate_fn

        Function that collate togethers images of different size for the dataloader
        It performs 
    """

    resizer = Resize(size)
    randomcrop = RandomResizedCrop(size=(size,size))
    mean = np.array([0.485, 0.456, 0.406]).mean()
    std = np.array([0.229, 0.224, 0.225]).sum()/np.sqrt(3)
    normalizer = Normalize(mean=mean, std=std)
    
    # Applying randomCrop
    images_tensor = []

    for x_ in x:
        image_tensor = torch.tensor(x_, dtype=torch.float32).unsqueeze(dim=0)
        image_tensor = resizer(image_tensor)
        image_tensor = randomcrop(image_tensor)

        images_tensor.append(image_tensor)

    output = torch.stack(images_tensor)
    output = normalizer(output/255.)

    return output