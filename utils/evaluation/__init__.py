"""
    This script contains the functions for evaluating the networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..image_processing import preprocess_image, numpy_to_tensor, tensor_to_numpy
from ..training import get_prediction
from ..image_processing import get_collater

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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
                if isinstance(decoded_image, tuple):
                    decoded_image = decoded_image[0]
                axs[i][j].imshow(tensor_to_numpy(decoded_image), cmap="gray")
                axs[i][j].set_title(f"AF {decoded_id+1}")

# Classifier evaluation

def get_prediction_dict(models_dict, X_test):
    y_hats = []

    # Getting predictions for each model
    for value in models_dict.values():
        y_hat = get_prediction(
            value,
            X_test,
            batch_size=1,
            collater_fn_x=get_collater(size=512, crop=False, resize_crop=False, rotate=False),
            use_gpu_if_available=True
        )

        y_hats.append(y_hat)

    predictions = dict(zip(models_dict.keys(), y_hats))

    return predictions

def get_metrics_array(predictions, y_trues, metrics=["Accuracy", "Recall", "Precision", "F1-Score"]):
    """
        This function an array of metrics according to predictions and y_trues
        The array is of size : (n_models, n_metrics)
    """
    
    metrics_functions = {
        "Accuracy":accuracy_score,
        "Recall":recall_score,
        "Precision":precision_score,
        "F1-Score":f1_score
    }

    performances_dict = {}
    for metric in metrics:
        if metric == "Accuracy" or y_trues.ndim == 1:
            if metric == "Accuracy":
                performances_dict[metric] = [(y_hat == y_trues).mean() for y_hat in predictions.values()]
            else:
                performances_dict[metric] = [metrics_functions[metric](y_hat, y_trues) for y_hat in predictions.values()]
        else:
            performances_dict[metric] = [metrics_functions[metric](y_hat, y_trues, average="micro", zero_division=0) for y_hat in predictions.values()]

    performances = np.array(list(zip(*list(performances_dict.values()))))
    
    return performances

def plot_performances(predictions, y_true, n_cols=3):
    # Parameters
    columns_names = [x.split("_")[-1] for x in y_true.columns]
    n_plots = len(columns_names) + 1
    n_rows = n_plots//n_cols + ((n_plots // n_cols) != 0)

    # Building plot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 10*n_rows))
    axs = axs.flatten()

    # Plotting
    metrics = ["Accuracy", "Recall", "Precision", "F1-Score"]

    performances_dict = {}

    for plot_name, plot_id in zip(["Global"]+columns_names, range(n_plots)):
        if plot_id != 0:
            predictions_ = dict([(key, value[:, plot_id-1])for key, value in predictions.items()])
            y_true_ = y_true.values[:, plot_id-1]
            
        else:
            predictions_ = predictions
            y_true_ = y_true.values

        # Getting data to plot
        performances = get_metrics_array(predictions_, y_true_, metrics=metrics)
        performances_dict[plot_name] = performances
        for value, label in zip(performances, predictions.keys()):
            axs[plot_id].scatter(metrics, value, label=label)
            axs[plot_id].legend()
            axs[plot_id].set_ylim(0, 1)
            axs[plot_id].set_title(plot_name)

        # Creating table
    table_index = predictions.keys()
    table = pd.concat(
        [
            pd.DataFrame(
                values, 
                columns=pd.MultiIndex.from_tuples(zip([key for i in range(len(key))], metrics)), 
                index=table_index) for key, values in performances_dict.items()
        ], axis=1)

    return table