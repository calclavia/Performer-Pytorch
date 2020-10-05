import torch
import torch.nn as nn
import math

def nonnegative_softmax_kernel_feature_creator(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        batch_dims_t,
        is_query: bool,
        normalize_data: bool=True,
        eps: float=0.0001):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      batch_dims_t: tuple of batch dimensions
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      normalize_data: predicate indicating whether data should be normalized,
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """
    if normalize_data:
        # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
        # w_norm = w * data_normalizer for w in {q,k}.
        data_normalizer = 1.0 / (math.sqrt(math.sqrt(data.shape[-1])))
    else:
        data_normalizer = 1.0

    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    data_thick_random_matrix = torch.zeros(data_mod_shape) + projection_matrix


    data_dash = torch.matmul(
        data_normalizer * data,
        data_thick_random_matrix.transpose(-1, -2)
    )

    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=data.ndim - 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = diag_data.unsqueeze(-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True)[0]) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash
