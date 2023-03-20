# Pytorch Point Utils
UNDER DEVELOPMENT
Pytorch Point Utils is a python library built using PyTorch that contains helper functions for matrix operations and transformations commonly used in computer vision and robotics.

## Installation
To install the library, move to the library directory and run the following command:

```bash
pip install .
```

## Example Usage


```python
import torch
from pytorch_point_utils import matrix_utils, transform, pointcloud

# Generate a batch of 3D points
points = torch.rand(100, 3)

points_jittered = pointcloud.jitter(points, sigma = 0.01, clip = 0.05)
points_rotated = pointcloud.random_rotate(points)
```

## Files
The library contains functions separated into three modules: matrix_utils, transform, and pointcloud


### matrix_utils
The matrix_utils.py file contains functions for generating skew matrices, identity matrices and batched identity matrices. The functions in this file are:

1. skew(r): Generates a skew matrix from the input tensor r of shape (..., 3).

2. eye_like(tf): Generates an identity matrix of the same shape and device as the input tensor tf of shape (..., 4, 4).

3. batch_eye(shape, device = None, dtype = None): Generates a batched identity matrix with the given shape, device, and dtype.

### transform
The transform.py file contains functions for transformations commonly used in computer vision and robotics. The functions in this file are:

1. inverse(tf): Computes the inverse of a transformation matrix tf of shape (..., 4, 4).

2. extract_translation(tf): Extracts the translation matrix from the input transformation matrix tf of shape (..., 4, 4).

3. extract_rotation(tf): Extracts the rotation matrix from the input transformation matrix tf of shape (..., 4, 4).

4. decompose(tf): Decomposes the input transformation matrix tf of shape (..., 4, 4) into rotation and translation matrices.

5. compose(trans, rot): Generates a transformation matrix from the input translation and rotation matrices trans and rot of shape (..., 4, 4).
...

### pointcloud
...