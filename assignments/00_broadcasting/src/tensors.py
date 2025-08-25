from __future__ import annotations

import numpy as np
import torch
from typing import Any


# 1. Tensor information
def get_info(
    T: torch.Tensor,
) -> tuple[int, Any, Any, Any]:

    # TODO: get the number of dimensions, the data type, and the shape of T, and the device that T is stored on
    pass


# 2. Slicing
def swap_matrix_quadrant(
    T: torch.Tensor,  # 4 x 4
) -> torch.Tensor:

    # TODO: Swap the top left and bottom right 2x2 matrix quadrants of T
    pass


# 3. Elementary and Elementwise Operations
def normalize_and_abs(
    T: torch.Tensor,  # N
    minval: float | torch.Tensor,  # scalar | N
    maxval: float | torch.Tensor,  # scalar | N
) -> torch.Tensor:

    # TODO: Normalize the entries of the sum of T1 and T2 between -1 and 1 based on the given minimum and maximum value, and then compute the absolute value of each element
    pass


# 4. Boolean Array Indexing
def replace_near_zero(
    T: torch.Tensor,
) -> torch.Tensor:

    # TODO: Replace all entries of T that are in [-1,1] with 0
    pass


# 5. Integer Array Indexing
def select_matrices_from_batch(
    T: torch.Tensor,  # B x N x M
    indices: list,  # K
) -> torch.tensor:  # K x N x M

    # TODO: Create a tensor of matrices from the batch of matrices given in T using the given indices
    pass


# 6. Tensor Generation and Data Type and Device Conversion
def generate_and_convert_tensors(
    data: np.array,  # unknown
    device: torch.device,
) -> tuple[
    torch.Tensor, # unknown
    torch.Tensor, # unknown
    torch.Tensor, # 4 x 4
    torch.Tensor, # unknown
    torch.Tensor, # unknown
]:  

    # TODO: create a tensor from the given numpy array
    pass
    # TODO: create a tensor containing only zeros of the same shape as the tensor created from the numpy array
    pass
    # TODO: create a tensor of size 4 x 4 filled with the value 42
    pass
    # TODO: convert the tensor created from the numpy array to 8-bit unsigned integers
    pass
    # TODO: move the 8-bit unsigned integer tensor to the given device
    pass


# 7. Dimensions as Arguments
def max_column_sum(
    T: torch.Tensor,  # N x M
) -> torch.Tensor:  # 1

    # TODO: Sum up the values of each column of T and then compute the maximum while making sure the resulting tensor is 1D with size 1
    pass


# 8. Concatenation and Stacking
def create_matrix_from_vectors(
    x: torch.Tensor,  # 2
    y: torch.Tensor,  # 2
    z: torch.Tensor,  # 2
) -> torch.Tensor:  # 2 x 3

    # TODO: create a matrix from the given vectors with the vectors as its columns
    pass


# 9. Reshape and View
def interleave_vectors(
    x: torch.Tensor,  # length
    y: torch.Tensor,  # length
) -> torch.Tensor:  # 2*length

    # TODO: Interleave the entries of the two vectors
    pass


# 10. Transposing Tensors
def transpose_matrices(
    T: torch.Tensor,  # N x M or B x N x M
) -> torch.Tensor:  # M x N or B x M x N

    # TODO: Transpose the given matrix or the given batch of matrices
    pass


# 11. Broadcasting and Singleton Dimensions
def make_broadcastable_1(
    T1: torch.Tensor,  # 2 x 3 x 2
    T2: torch.Tensor,  # 2 x 3
) -> tuple[
    torch.Tensor, # unknown
    torch.Tensor, # unknown
]:

    # TODO: make the tensors broadcastable by using singleton dimensions
    pass


# 11. Broadcasting and Singleton Dimensions
def make_broadcastable_2(
    T1: torch.Tensor,  # 2 x 3 x 4 x 5 x 1
    T2: torch.Tensor,  # 2 x 4 x 1
) -> tuple[
    torch.Tensor, # unknown
    torch.Tensor, # unknown
]:

    # TODO: make the tensors broadcastable by using singleton dimensions
    pass


# 11. Broadcasting and Singleton Dimensions
def make_broadcastable_3(
    T1: torch.Tensor,  # 3 x 1 x 2
    T2: torch.Tensor,  # 3 x 2 x 7
) -> tuple[
    torch.Tensor, # unknown
    torch.Tensor, # unknown
]:

    # TODO: make the tensors broadcastable by using singleton dimensions
    pass


# 11. Broadcasting and Singleton Dimensions
def batch_scalar_product(
    T1: torch.Tensor,  # batchsize x 3
    T2: torch.Tensor,  # batchsize x 3
) -> torch.Tensor:  # batchsize

    # TODO: compute the batched dot product of the two batches of vectors by using singleton dimensions and torch.bmm
    pass
