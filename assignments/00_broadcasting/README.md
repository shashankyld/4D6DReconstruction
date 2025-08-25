# Voluntary Exercise - Tensor Operations and Broadcasting

Over the course of the next exercise sheets, you will need to work with Pytorch using tensors. The aim of this exercise sheet is for you to get to know the basics of tensor operations, especially regarding broadcasting.

**This exercise sheet is completely voluntary. It will not be graded or count towards exam admission.** You may solve the whole sheet, only work on the parts that are interesting to you, or skip it entirely.

Hints are provided to help you solve the exercises, or just to give you a general idea of how to work with tensors. Even in the case that you do not want to solve the exercises, these hints might still be helpful to you.

At the end of the exercise sheet you will find additional remarks regarding solving this exercise sheet.


## Task

In `src/tensors.py`, complete the following exercises:

1. **Tensor Information**: Implement the function `get_info` to return the number of dimensions, the data type, the shape and the device of a given tensor.

    **Hint**: Tensors have different properties that are helpful to know, e.g. for debugging purposes. Mismatches in the [number of dimensions](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.ndim.html), the [data type](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype) and the [shape](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.shape.html) of tensors can easily lead to errors when running the code. Furthermore, as Pytorch can utilize GPUs for parallel computations, two tensors need to be stored on the same [device](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.device.html) (CPU or one of the available GPUs) if you want to perform an operation using both.
2. **Slicing**: Implement the function`swap_matrix_quadrant` that, given a tensor of shape 4x4, swaps its quadrants of size 2x2 as follows, and returns the result as a new tensor, without modifying the input tensor.

    ```
        A = ⌈ A11 A12 ⌉ -> ⌈ A22 A12 ⌉
            ⌊ A21 A22 ⌋    ⌊ A21 A11 ⌋
    ```
    **Hint**: To access the `i`-th element of a tensor `T` in its first dimension, one uses `T[i]`. Negative indices reference the tensor from back to front. To get a slice instead of a single element, one uses `T[start:end:step]`, where omitting one of the values uses the defaults 0, the size in the indexed dimension, and 1. A negative `step` swaps the order of the returned values.

    **Hint**: To make an explicit copy of a tensor, use [torch.Tensor.clone](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.clone.html). Simply assigning a tensor to a new variable and modifying any of the two will also modify the other tensor as they reference the same memory, which can lead to unwanted behavior.
3. **Elementary and Elementwise Operations**: Implement the function `normalize_and_abs` that first normalizes some values in a tensor given a minimum and a maximum to `[-1, 1]`, and then computes the absolute value for each normalized entry and returns it.

    **Hint**: Pytorch supports elementary operations such as `+`, `-`, `*`, `/` etc. between two tensors of the same size or via broadcasting, and also between a tensor and a scalar value, which are applied as elementwise operations. Other elementwise operations such as [abs](https://docs.pytorch.org/docs/stable/generated/torch.abs.html), [cos](https://docs.pytorch.org/docs/stable/generated/torch.cos.html), [exp](https://docs.pytorch.org/docs/stable/generated/torch.exp.html) etc. can also applied to a single tensor without changing the tensor's shape.
4. **Boolean Array Indexing**: Implement the function `replace_near_zero` that replaces any tensor entries of a given tensor, that are in the interval [-1, 1], with 0.
    
    **Hint**: Tensors containing booleans (masks) can be utilized to index only elements of a tensor that fulfil certain conditions. Passing a mask as an index to another tensor selects the elements that are `True` in the mask. Both tensors need to have the same shape. A mask can be created e.g. by comparing a tensor to a scalar.

    **Hint**: For elementwise boolean operations e.g. [torch.logical_and](https://docs.pytorch.org/docs/stable/generated/torch.logical_and.html) and [torch.logical_or](https://docs.pytorch.org/docs/stable/generated/torch.logical_or.html) can be used. [torch.logical_not](https://docs.pytorch.org/docs/stable/generated/torch.logical_not.html) or `~` inverts a boolean tensor.
5. **Integer Array Indexing**: Implement the function `select_matrices_from_batch` that receives a batch of 2D matrices and an array of indices, and returns a tensor with only the 2D matrices selected by the indices.

    **Hint**: To select a new tensor from a given tensor that only contains entries at specific position one can pass an integer array as index. Tensors with multiple dimensions require separate indices per dimension, where the indexing tensors need to be broadcastable and can only contain indices within the bounds of the tensor. Different indexing methods (slicing, masking, ...) can be combined.

    Example:

    ```
    x = torch.tensor([10, 20, 30, 40, 50])        x[[1, 3]] -> [20, 40]
    y = torch.tensor([[1, 2], [3, 4], [5, 6]])    y[[0, 2], [1, 0]] -> [2, 5]
    ```
6. **Tensor Generation and Data Type and Device Conversion**: Implement the function `generate_and_convert_tensors` that performs the following set of tasks:
    1. Create a tensor from the given numpy array
    2. Create a tensor containing only zeros with the same shape as the tensor created from the numpy array
    3. Create a tensor of size 4x4 that is filled with the integer value 42
    4. Convert the tensor created from the numpy array to 8-bit unsigned integers
    5. Move the 8-bit unsigned integer tensor to the given device

    **Hint**: There are several option to create new tensors, including 
    - [torch.from_numpy](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html): generate a tensor from a given numpy array, the memory is shared
    - [torch.tensor](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html): generate an independent tensor from a numpy array, a list or a tuple
    - [torch.rand](https://docs.pytorch.org/docs/stable/generated/torch.rand.html): generate a tensor of given size with uniformly distributed random values between 0 and 1
    - [torch.zeros](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html): generate a tensor full of zeros with given size
    - [torch.zeros_like](https://docs.pytorch.org/docs/stable/generated/torch.zeros_like.html): generate a tensor full of zeros with the size of the given tensor
    - [torch.ones](https://docs.pytorch.org/docs/stable/generated/torch.ones.html): generate a tensor full of ones with given size
    - [torch.full](https://docs.pytorch.org/docs/stable/generated/torch.full.html): generate a tensor of given size full of the given value

    **Hint**: Sometimes, a different data type is required for a tensor, e.g. `torch.float32` instead of `torch.uint8` or it needs to be moved to a different device so it can be used. This is performed with [torch.Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html) in both cases.
7. **Dimensions as Arguments**: Implement the function `max_column_sum` that computes the sum of all the values in each column of a 2D tensor and then returns the maximum of the sums as a 1D tensor (not a scalar).

    **Hint**: Some functions accept the dimension as an optional argument `dim`, so that an operation is only performed along that axis, e.g. [torch.sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html), [torch.min](https://docs.pytorch.org/docs/stable/generated/torch.min.html), [torch.amin](https://docs.pytorch.org/docs/stable/generated/torch.amin.html), [torch.max](https://docs.pytorch.org/docs/stable/generated/torch.max.html) or [torch.amax](https://docs.pytorch.org/docs/stable/generated/torch.amax.html). It is possible to pass a list or tuple to perform the operation across multiple dimensions. The boolean argument `keepdim` determines if the operation should reduce the number of dimensions or if the corresponding dimensions should be kept.
8. **Concatenation and Stacking**: Implement the function `create_matrix_from_vectors` that creates a matrix from three given vectors, that are used as the columns of the matrix in the order they are provided to the function.

    **Hint**: To merge several tensors, you have two options:
    - You can use [torch.cat](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) to concatenate them along an existing axis. In this case, the shape of the tensors must be the same, apart from the dimension used for concatenation.
    - You can use [torch.stack](https://docs.pytorch.org/docs/stable/generated/torch.stack.html) to stack them along a new axis. In this case, all tensors must have the exact same shape, and the new dimension is added at the given position (default is 0).
9. **View and Reshape**: Implement the function `interleave_vectors` that takes two 1D vectors of the same length and returns a 1D vector that contains the entries of both tensors in an alternating order as follows:

    ```
    x = [x1, x2, ..., xn], y = [y1, y2, ..., yn] -> [x1, y1, x2, y2, ..., xn, yn]
    ```
    **Hint**: To represent a tensor in a different shape, [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) returns a "view" of a tensor with the desired shape, which references the same memory as the input tensor. [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html) returns a view of the tensor if possible, else it returns a copy of the tensor. In both cases, the original and new shape need to have the same size (number of elements). Setting a single dimension to `-1` infers its size from the remaining dimensions and the size of the input tensor.
10. **Transposing**: Implement the function `transpose_matrices` that receives either a 2D matrix or a 3D tensor containing a batch of 2D matrices and transposes the 2D matrix or all 2D matrices, respectively.

    **Hint**: Using [torch.transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html), any two dimensions of a tensor can be transposed, while [torch.t](https://docs.pytorch.org/docs/stable/generated/torch.t.html) only works on 2D tensors, and leaves 0D and 1D tensors unchanged.
11. **Broadcasting and Singleton Dimensions**:

    (a) Implement the functions `make_broadcastable_1`, `make_broadcastable_2`, and `make_broadcastable_3` that get two tensors as input and adjust the dimensions of the tensors in a way that they are broadcastable when returned and can be added.

    (b) Implement the function `batch_scalar_product` that computes the scalar products of pairs of vectors from two batches (for the `i`-th scalar product, vector `i` in batch 1 and vector `i` in batch two are used). Use [torch.bmm](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) for the computation. 

    **Hint**: Broadcasting follows a [set of rules](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) that determines, if two tensors are broadcastable (automatically expanded to have the same size). These are:
    - Each tensor needs to have at least one dimension.
    - When iterating over the dimension sizes starting at the last dimension, the dimension sizes must either
        * be equal,
        * one of them is 1, or
        * one of them does not exist.

    As an example, a tensor of size `torch.Size([5,1,4,1])` and a second tensor of size `torch.Size([3,1,1])` are broadcastable:
    - The last dimension is equal (1),
    - the second to last dimension is 4 and 1 in the respective tensors, of which one is 1,
    - the third to last dimension is 1 and 3 in the respective tensors, of which one is 1, and
    - the forth to last dimension (which is the first) does not exist for the second tensor.

    The resulting tensor would be of shape `torch.Size([5,3,4,1])`.

    **Hint**: Sometimes, a singleton dimension (of size 1) needs to be added or deleted to use broadcasting with two tensors.
    - [torch.unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html) adds a new dimension at the given position. Indexing a dimension of a tensor as `None` does the same. Furthermore, [torch.Tensor.expand](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html) and [torch.broadcast_to](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html) can be used to broadcast a tensor to a given shape and return a view of the tensor. If increasing the number of dimensions like this, it will add new dimensions in the front. [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html) and [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) are additional options to add dimensions.
    - [torch.squeeze](https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html) removes a dimension at a given position, if the size of the dimension is 1, or removes all dimensions of size 1 without a given dimension. Alternatively, indexing the respective dimension with 0, as well as [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html) and [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) can be used.

    **Hint**: Other options to multiply two matrices are [torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) or the `@`-operator, which offer more options on how two matrices can be combined, depending on their dimensionalities. [torch.bmm](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) requires both matrices to be 3-dimensional.


## General Remarks

This exercise sheet is voluntary and will not be graded. A set of unit tests is provided for you to check your solutions. To run them, use 

```
nox -s tests
```

In the case that you only want to solve a subset of the tasks from this exercise sheet, you may adjust the variable `tasks_solved` in `tests/test_tensors.py`: Deleting an exercise number from the list will also exclude the corresponding tests when you run the test. This way, the tests for exercises you did not solve will not fail.

After completing the tasks, you should see the result in `demo_output.md` when running the `demo.py` example. In the `demo.py` file, you will also find the `tasks_solved` variable. If you are interested in only the outputs of a subset of exercises, you may also remove numbers of unsolved tasks to only generate the output relevant to you.

As an additional remark: When solving the exercises and in general when working with Pytorch, it can be very helpful to know the shapes of the tensors you are working with (especially when revisiting code after some time or for debugging), e.g. as comments after any operations applied to the tensors:

```python
A = torch.rand(1000,3) # batch_dim x 3
```

<br/>
<center><h3>Good Luck!</h3></center>
