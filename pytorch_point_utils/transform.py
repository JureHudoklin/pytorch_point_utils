import torch

from .matrix_utils import skew, batch_eye

def inverse(tf: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a transformation matrix.

    Parameters
    ----------
    tf : torch.Tensor # (..., 4, 4)

    Returns
    -------
    tf_inv : torch.Tensor # (..., 4, 4)
    """
    tf_inv = torch.zeros_like(tf)

    rot = tf[..., :3, :3].clone()
    trans = tf[..., :3, 3:]
    rot_inv = torch.transpose(rot, -2, -1)

    tf_inv[..., :3, :3] = rot_inv
    tf_inv[..., :3, 3:] = -1.0 * torch.matmul(rot_inv, trans)
    tf_inv[..., 3, 3] = 1.0

    return tf_inv

def extract_translation(tf) -> torch.Tensor:
    """
    Extract translation matrix from transformation matrix
    
    Parameters
    ----------
    tf : torch.Tensor # (..., 4, 4)
    
    Returns
    -------
    torch.Tensor # (..., 4, 4)
    """
    
    trans = tf.clone()
    trans[..., :3, :3] = torch.eye(3, device=tf.device, dtype=tf.dtype)
    
    return trans
    
def extract_rotation(tf) -> torch.Tensor:
    """
    Extract rotation matrix from transformation matrix
    
    Parameters
    ----------
    tf : torch.Tensor # (..., 4, 4)
    
    Returns
    -------
    torch.Tensor # (..., 4, 4)
    """
    
    rot = tf.clone()
    rot[..., :3, 3] = 0
    
    return rot

def decompose(tf) -> torch.Tensor:
    """
    Decompose transformation matrix into rotation and translation
    
    Parameters
    ----------
    tf : torch.Tensor # (..., 4, 4)
    
    Returns
    -------
    torch.Tensor # (..., 4, 4)
    torch.Tensor # (..., 4, 4)
    """
    return extract_rotation(tf), extract_translation(tf)

def compose(trans: torch.tensor, rot:torch.tensor) -> torch.Tensor:
    """
    Generate transformation matrix from rotation and translation

    Parameters
    ----------
    trans : torch.tensor # (..., 4, 4)
    rot : torch.tensor # (..., 4, 4)
    
    Returns
    -------
    tf : torch.tensor # (..., 4, 4)
    """
    
    tf = torch.matmul(trans, rot)
    
    return tf

def rotation_from_vector(v: torch.Tensor) -> torch.Tensor:
    """
    Convert a vector to a rotation matrix using Rodrigues' formula

    Parameters
    ----------
    v : torch.Tensor # (..., 3)

    Returns
    -------
    rot : torch.Tensor # (..., 4, 4)
    """
    one_dim = False
    if v.dim() == 1:
        v = v.unsqueeze(0)
        one_dim = True
        
    v = v.unsqueeze(0) if v.dim() == 1 else v
    theta = torch.norm(v, dim=-1, keepdim=True).repeat(1, 3)
    theta = theta + 1e-8
    theta_un = theta.unsqueeze(-1).repeat(1, 1, 3)
    v = v / theta # (B, 3)
    r1 = batch_eye(v.shape, device=v.device, dtype=v.dtype)
    r2 = torch.sin(theta_un) * skew(v)
    r3 = (1 - torch.cos(theta_un)) * torch.matmul(skew(v), skew(v))
    
    rot = r1 + r2 + r3
    
    rot_out = torch.eye(4, device=rot.device, dtype=rot.dtype).unsqueeze(0).repeat(rot.shape[0], 1, 1)
    rot_out[:, :3, :3] = rot
    rot_out = rot_out.squeeze(0) if one_dim else rot_out
    
    return rot_out

def get_random_rotation(shape=None, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    Generate a random rotation matrix

    Parameters
    ----------
    shape : tuple, optional
    device : str, optional
    dtype : torch.dtype, optional

    Returns
    -------
    rot : torch.Tensor # (..., 4, 4)
    """
    if shape is None:
        rot = torch.rand(3, device=device, dtype=dtype)*3.14
    else:
        rot = torch.randn(*shape, 3, device=device, dtype=dtype)*3.14
    
    rot = rotation_from_vector(rot)
    return rot

def get_random_translation(shape=None, device='cpu', dtype=torch.float32, translation_range=1.0) -> torch.Tensor:
    """
    Generate a random translation matrix

    Parameters
    ----------
    shape : tuple, optional
    device : str, optional
    dtype : torch.dtype, optional
    translation_range : float, optional
    
    Returns
    -------
    trans : torch.Tensor # (..., 4, 4)
    """
    
    if shape is None:
        trans = torch.eye(4, device=device, dtype=dtype)
        trans[:3, 3] = torch.randn(3, device=device, dtype=dtype)*translation_range
    else:
        shape = shape if isinstance(shape, tuple) else (shape,)
        trans = batch_eye([*shape, 4], device=device, dtype=dtype)
        trans[..., :3, 3] = torch.randn(*shape, 3, device=device, dtype=dtype)*translation_range
        
    return trans

def get_random_transform(shape=None, device='cpu', dtype=torch.float32, translation_range=1.0) -> torch.Tensor:
    """
    Generate a random transformation matrix

    Parameters
    ----------
    shape : tuple, optional
    device : str, optional
    dtype : torch.dtype, optional
    translation_range : float, optional
    
    Returns
    -------
    tf : torch.Tensor # (..., 4, 4)
    """
    tf = torch.matmul(get_random_translation(shape, device, dtype, translation_range), 
                      get_random_rotation(shape, device, dtype))
    return tf
