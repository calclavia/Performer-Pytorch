from performer.jax_kernel import nonnegative_softmax_kernel_feature_creator as jax_kernel
from performer.kernel import nonnegative_softmax_kernel_feature_creator as kernel

import unittest

import torch
import numpy as np


class TestPerformer(unittest.TestCase):
    def test_kernel(self):
        for is_query in [False, True]:
            # Batch, seq len, d
            data = np.random.rand(2, 4, 8)
            # Batch, r, d
            projection_matrix = np.random.rand(3, 8)

            jax_output = jax_kernel(
                data,
                projection_matrix,
                attention_dims_t=None,
                batch_dims_t=[0],
                precision=None,
                is_query=is_query
            )
            print('jax_output', jax_output.shape)
            print()

            output = kernel(
                torch.from_numpy(data),
                torch.from_numpy(projection_matrix),
                batch_dims_t=[0],
                is_query=is_query
            )
            print('output', output.shape)
            assert jax_output.shape == output.shape, (jax_output.shape, output.shape)
            assert np.allclose(jax_output, output)

if __name__ == '__main__':
    unittest.main()
