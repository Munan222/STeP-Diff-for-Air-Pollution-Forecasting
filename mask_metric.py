import torch
import numpy as np


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    mask = (labels > mask_value).float() 

    # Calculate the loss
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask  # Apply the mask
    return torch.mean(loss[mask > 0]) 




def masked_mae(preds, labels, mask_value):
    # Convert NumPy arrays to PyTorch tensors
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)

    # Create a mask for labels that are greater than 0
    mask = (labels > mask_value).float()  # Only consider labels that are not zero

    # Calculate masked MAE
    mae = (mask * (preds - labels).abs()).sum() / mask.sum()
    return mae.item()  # Return as a Python float

def masked_mse(preds: torch.Tensor, labels: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    mask = (labels > mask_value).float() 

    # Calculate the loss
    loss = (preds - labels) ** 2
    loss = loss * mask  # Apply the mask
    return torch.mean(loss[mask > 0])


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """root mean squared error.
    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.
    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(preds=preds, labels=labels, mask_value=mask_value))
