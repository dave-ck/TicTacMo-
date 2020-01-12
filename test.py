import numpy as np
import torch as T

print("A commit - this needs reverting.")
def check_nan(tensor_or_array):
    if type(tensor_or_array) is np.ndarray:
        if np.isnan(tensor_or_array).any():
            raise ValueError("NDArray contains nan:", tensor_or_array)
    elif type(tensor_or_array) is T.Tensor:
        if T.isnan(tensor_or_array).any():
            raise ValueError("Tensor contains nan:", tensor_or_array)
    else:
        raise ValueError("Not Tensor or Array:", tensor_or_array)
