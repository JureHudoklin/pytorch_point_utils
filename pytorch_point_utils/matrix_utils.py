import torch

def skew(r):
    """Generate a skew matrix

    Parameters
    ----------
    r : torch.Tensor # (..., 3)

    Returns
    -------
    sk : torch.Tensor # (..., 3, 3)
    """
    sk = torch.zeros((*r.shape[:-1], 3, 3), device=r.device, dtype=r.dtype)
    sk[..., 0, 1] = -r[..., 2]
    sk[..., 0, 2] = r[..., 1]
    sk[..., 1, 0] = r[..., 2]
    sk[..., 1, 2] = -r[..., 0]
    sk[..., 2, 0] = -r[..., 1]
    sk[..., 2, 1] = r[..., 0]
    return sk

def eye_like(tf):
    """
    Parameters
    ----------
    tf : torch.Tensor # (..., 4, 4)
    
    Returns
    -------
    torch.Tensor # (..., 4, 4)
    """
    return torch.eye(4, device=tf.device, dtype=tf.dtype).unsqueeze(0).repeat(tf.shape[0], 1, 1)

def batch_eye(shape, device = None, dtype = None):
    """ Generate a batched identity matrix

    Parameters
    ----------
    shape : list # (..., N)

    Returns
    -------
    batched_eye : torch.Tensor # (..., N, N)
    """
    batched_eye = torch.eye(shape[-1], device=device, dtype=dtype)
    batched_eye = batched_eye.unsqueeze(0).repeat(*shape[:-1], 1, 1)
    return batched_eye

