
import abc
from collections.abc import Iterable  # pylint: disable=g-importing-member
import functools
from absl import logging
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import numpy as onp


def nonnegative_softmax_kernel_feature_creator(data,
                                               projection_matrix,
                                               attention_dims_t,
                                               batch_dims_t,
                                               precision,
                                               is_query,
                                               normalize_data=True,
                                               eps=0.0001):
    """Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      attention_dims_t: tuple of attention dimensions
      batch_dims_t: tuple of batch dimensions
      precision: precision parameter
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      normalize_data: predicate indicating whether data should be normalized,
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """
    del attention_dims_t
    if normalize_data:
        # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
        # w_norm = w * data_normalizer for w in {q,k}.
        data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
    else:
        data_normalizer = 1.0
    ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

    # print('data', data.shape)
    # print('data_thick_random_matrix', data_thick_random_matrix.shape)

    data_dash = lax.dot_general(
        data_normalizer * data,
        data_thick_random_matrix,
        (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
         (batch_dims_t, batch_dims_t)),
        precision=precision)

    # print('data_dash', data_dash.shape)

    diag_data = jnp.square(data)
    diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    # print('diag_data', diag_data.shape)
    diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)
    # print('diag_data', diag_data.shape)


    if is_query:
        last_dims_t = (len(data_dash.shape) - 1,)
        data_dash = ratio * (
            jnp.exp(data_dash - diag_data -
                    jnp.max(data_dash, axis=last_dims_t, keepdims=True)) + eps)
    else:
        data_dash = ratio * (
            jnp.exp(data_dash - diag_data - jnp.max(data_dash)) + eps)

    return data_dash
