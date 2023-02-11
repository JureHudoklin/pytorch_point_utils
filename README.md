# Pytorch Point Utils

Pytorch Point Utils is a python library built using PyTorch that contains helper functions for matrix operations and transformations commonly used in computer vision and robotics.

## Installation
To install the library, run the following command:

```bash
pip install .
```

## Files
The library contains functions separated into three files: matrix_utils.py, transform.py, and pointcloud.py

### matrix_utils.py
The matrix_utils.py file contains functions for generating skew matrices, identity matrices and batched identity matrices. The functions in this file are:

1. skew(r): Generates a skew matrix from the input tensor r of shape (..., 3).

2. eye_like(tf): Generates an identity matrix of the same shape and device as the input tensor tf of shape (..., 4, 4).

3. batch_eye(shape, device = None, dtype = None): Generates a batched identity matrix with the given shape, device, and dtype.

### transform.py
The transform.py file contains functions for transformations commonly used in computer vision and robotics. The functions in this file are:

1. inverse(tf): Computes the inverse of a transformation matrix tf of shape (..., 4, 4).

2. extract_translation(tf): Extracts the translation matrix from the input transformation matrix tf of shape (..., 4, 4).

3. extract_rotation(tf): Extracts the rotation matrix from the input transformation matrix tf of shape (..., 4, 4).

4. decompose(tf): Decomposes the input transformation matrix tf of shape (..., 4, 4) into rotation and translation matrices.

5. compose(trans, rot): Generates a transformation matrix from the input translation and rotation matrices trans and rot of shape (..., 4, 4).
...

### pointcloud.py
...