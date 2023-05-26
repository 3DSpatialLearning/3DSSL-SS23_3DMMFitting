import torch
import numpy as np

def to_device(data: dict[str, np.array], device: str) -> dict[str, torch.tensor]:
    for k, v in data.items():
        data[k] = v.to(device)
    return data

def dict_tensor_to_np(data: dict[str, torch.tensor]) -> dict[str, np.array]:
    for k, v in data.items():
        data[k] = v.detach().cpu().squeeze().numpy()
    return data