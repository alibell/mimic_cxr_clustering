import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

def train_model (model, X, y, n_epoch, batch_size=4, collater_fn_x=None, collater_fn_y=None, use_gpu_if_available=True, loss_names=None, shuffle=True):
    """
        This function perform the model training and dynamically display the loss

        Parameters:
        -----------
        model: model object to train, should contains a fit method which returns the loss values as a list
        X: X dataset set or tuple of X, each one should be sliceable
        y: y dataset or tuple of y, each one should be sliceable
        n_epoch: int, number of epochs
        batch_size: int, batch size
        collater_fn: function, if not None, the collater function is applied to the Xs
        use_gpu_if_available: boolean, if True the gpu is used if available
        shuffle: boolean, if True the data are shuffled
    """

    # Setting parameters
    if use_gpu_if_available and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    n_samples = len(X)
    model = model.to(device)

    dataloader = DataLoader(list(range(n_samples)), shuffle=shuffle, batch_size=batch_size)

    for epoch in range(n_epoch):

        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            losses = None

            for idx in tepoch:
                # Filtering data
                batch_data = []
                for data, collater_fn in zip([X, y], [collater_fn_x, collater_fn_y]):
                    # Setting collater to identity is None
                    if collater_fn is None:
                        collater_fn_ = lambda x: x
                    else:
                        collater_fn_ = collater_fn

                    if data is not None:
                        if isinstance(data, tuple):
                            data_ = tuple([collater_fn_(x_[idx]).to(device) for x_ in data])
                        else:
                            data_ = collater_fn_(data[idx]).to(device)
                    else:
                        data_ = None
                        
                    batch_data.append(data_)

                # Fitting and getting loss
                loss = model.fit(batch_data[0], batch_data[1])
                if loss_names is not None:
                    loss = dict(zip(loss_names, loss))
                else:
                    loss = dict(zip([f"Loss {l}" for l in range(len(loss))],loss))


                # Storing loss for mean loss calculation
                if losses is None:
                    losses = dict([(key, [value]) for key, value in loss.items()])
                else:
                    for key, value in loss.items():
                        losses[key].append(value)

                # Getting mean loss
                mean_losses = dict([(f"Mean {key}", np.array(value).mean()) for key, value in losses.items()])

                # Displaying loss
                tepoch.set_postfix(**loss, **mean_losses)

def get_prediction (model, X, batch_size=4, collater_fn_x=None, use_gpu_if_available=True):
    """
        This function get the model predictions

        Parameters:
        -----------
        model: model object to use for prediction, should contains a predict method which returns the predictions
        X: X dataset set or tuple of X, each one should be sliceable
        batch_size: int, batch size
        collater_fn: function, if not None, the collater function is applied to the Xs
        use_gpu_if_available: boolean, if True the gpu is used if available
    """

    # Setting parameters
    if use_gpu_if_available and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    n_samples = len(X)
    model = model.to(device)

    dataloader = DataLoader(list(range(n_samples)), shuffle=False, batch_size=batch_size)
    y_hats = []

    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Predictions")
        losses = None

        for idx in tepoch:
            # Filtering data
            batch_data = []
            for data, collater_fn in zip([X], [collater_fn_x]):
                # Setting collater to identity is None
                if collater_fn is None:
                    collater_fn_ = lambda x: x
                else:
                    collater_fn_ = collater_fn

                if data is not None:
                    if isinstance(data, tuple):
                        data_ = tuple([collater_fn_(x_[idx]).to(device) for x_ in data])
                    else:
                        data_ = collater_fn_(data[idx]).to(device)
                else:
                    data_ = None
                    
                batch_data.append(data_)

            # Fitting and getting loss
            with torch.no_grad():
                y_hat = model.predict(batch_data)

            y_hats.append(y_hat)

    y_hats = np.concatenate(y_hats, axis=0)

    return y_hats