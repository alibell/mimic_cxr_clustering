"""
    This script contains the functions for evaluating the networks
"""

import numpy as np
import matplotlib.pyplot as plt
from ..image_processing import preprocess_image, numpy_to_tensor, tensor_to_numpy

# Images evaluations

def display_predictions(dataset, models, device="cpu"):
    """
        This function display the AE reconstruction given a list of images, it generate randomly a list of images to evaluate

        Parameters:
        -----------
        dataset: image dataset object
        models: [Auto Encoder object], list of auto-encoders to evaluate containing a predict function
        device: str, device to use for prediction
    """

    # Images IDs
    images_ids = np.random.choice(range(len(dataset)), 3)

    # Models to device
    models = [x.to(device) for x in models]

    has_noisy = "generate_noisy_image" in dir(models[0])
    n_images = 1+has_noisy+len(models)

    # Creating the image
    fig, axs = plt.subplots(len(images_ids), n_images, figsize=(6*n_images,15))

    for i in range(len(images_ids)):
        image = dataset[[images_ids[i]]][0]
        preprocessed_image = preprocess_image(numpy_to_tensor(image), size=512).to(device)
        input_image = preprocessed_image

        if has_noisy:
            noisy_preprocessed_image = models[0].generate_noisy_image(preprocessed_image, variance=models[0].noise_variance, p_salt_pepper=models[0].p_salt_pepper_noise)
            input_image = noisy_preprocessed_image

        decoded_images = [model.predict(input_image.unsqueeze(0)) for model in models]

        for j in range(n_images):
            if j == 0:
                axs[i][j].imshow(tensor_to_numpy(preprocessed_image), cmap="gray")
                axs[i][j].set_title("Original")
            elif j == 1 and has_noisy:
                axs[i][j].imshow(tensor_to_numpy(noisy_preprocessed_image), cmap="gray")      
                axs[i][j].set_title("Original with gaussian noise")
            else:
                decoded_id = j-(1+has_noisy)
                decoded_image = decoded_images[decoded_id]
                if isinstance(decoded_image, list):
                    decoded_image = decoded_image[0]
                axs[i][j].imshow(tensor_to_numpy(decoded_image), cmap="gray")
                axs[i][j].set_title(f"AF {decoded_id}")
