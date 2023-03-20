import torch
from typing import Union

from .matrix_utils import skew, eye_like, batch_eye
from .transform import rotation_from_vector, inverse

def transform(points: torch.tensor, matrix: torch.tensor,
                 translate=True, rotate=True) -> torch.Tensor:
    """
    Transform points

    Parameters
    ----------
    points : torch.Tensor # (..., N, 3)
    matrix : torch.Tensor # (..., 4, 4)
    
    Returns
    -------
    torch.Tensor # (..., N, 3)
    """
    if not translate:
        matrix = matrix.clone()
        matrix[..., 3, :3] = 0
    if not rotate:
        matrix = matrix.clone()
        matrix[..., :3, :3] = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    point_new = points.clone()
    ones = torch.ones_like(points[..., :1])
    point_new = torch.cat([point_new, ones], dim=-1) # (B, N, 4)
    
    point_new = torch.matmul(matrix, point_new.transpose(-1, -2)) # (B, 4, 4) * (B, 4, N) -> (B, 4, N)
    point_new = point_new.transpose(-1, -2) # (B, N, 4)
    
    return point_new[..., :3]

def random_rotate(pc: torch.tensor) -> torch.tensor:
    """
    Parameters
    ----------
    pc : torch.Tensor # (..., N, 3)
    
    Returns
    -------
    torch.Tensor # (..., N, 3)
    """
    d2 = False
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
        d2 = True
        
    rotation_vec = torch.randn(pc.shape[0], 3, device=pc.device, dtype=pc.dtype) * 3.1415926
    rot = rotation_from_vector(rotation_vec)
    
    if d2:
        rot = rot.squeeze(0)
        pc = pc.squeeze(0)
            
    return transform(pc, rot), rot

def random_translate(pc: torch.tensor) -> torch.tensor:
    """
    Parameters
    ----------
    pc : torch.Tensor # (..., N, 3)
    
    Returns
    -------
    torch.Tensor # (..., N, 3)
    """
    d2 = False
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
        d2 = True
    
    trans = torch.randn(pc.shape[0], 3, device=pc.device, dtype=pc.dtype)
    
    trans_mat = torch.eye(4, device=pc.device, dtype=pc.dtype).unsqueeze(0).repeat(pc.shape[0], 1, 1)
    trans_mat[..., :3, 3] = trans.unsqueeze(-1)
    
    if d2:
        trans_mat = trans_mat.squeeze(0)
        pc = pc.squeeze(0)
    
    return transform(pc, trans_mat), trans_mat


def normalize(pc: torch.tensor, scale = 1.0, return_inverse: bool=False) -> Union[torch.tensor, torch.tensor]:
    """
    Normalizes the point cloud to have a pc mean of 0
    and a distance between between closest points with a standard deviation of 1.

    Parameters
    ----------
    pc : torch.Tensor # (..., N, 3)
    return_inverse : bool, optional
        Whether to return the inverse transformation matrix, by default False

    Returns
    -------
    pc : torch.tensor # (..., N, 3)
    tf : torch.tensor # (..., 4, 4)
    """
    d2 = False
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
        d2 = True
    
    trans = torch.eye(4, device=pc.device, dtype=pc.dtype).unsqueeze(0).repeat(pc.shape[0], 1, 1)
    trans[..., :3, 3] = -pc.mean(dim=-2) # (..., 3)
    
    avg_point_distance = knn(pc, pc, 2)[0] # (..., N, 2)
    avg_point_distance = avg_point_distance[..., 1].mean(dim=-1, keepdim=True) # (..., 1)
    
    scale_mtx = torch.eye(4, device=pc.device, dtype=pc.dtype).unsqueeze(0).repeat(pc.shape[0], 1, 1)
    scale_mtx = scale_mtx * scale #avg_point_distance # (..., 4, 4)
    scale_mtx[..., 3, 3] = 1

    tf = torch.matmul(scale_mtx, trans)
    pc = transform(pc, tf)
    
    if d2:
        tf = tf.squeeze(0)
        pc = pc.squeeze(0)
        
    if return_inverse:
        inv_tf = torch.linalg.inv(tf)
        return pc, inv_tf
    
    return pc, tf

def knn(src, tgt, k) -> torch.tensor:
    """
    Find the k nearest neighbors of each point in src.
    
    Parameters
    ----------
    src : torch.Tensor (..., N, C)
    tgt : torch.Tensor (..., M, C)
    k: int
    
    Returns
    -------
    val : torch.Tensor (..., N, k)
    idx : torch.Tensor (..., N, k)
    """
    src = src.unsqueeze(0) if src.dim() == 2 else src
    tgt = tgt.unsqueeze(0) if tgt.dim() == 2 else tgt
    
    dist = src.unsqueeze(-2) - tgt.unsqueeze(-3) # (..., N, M, C)
    dist = torch.sum(dist ** 2, dim=-1) # (..., N, M)
    val, idx = dist.topk(k, dim=-1, largest=False) # (..., N, k)
    
    val = torch.sqrt(val)
    
    return val, idx

def jitter(pc: torch.Tensor, sigma:int = 0.0005, clip:int = 0.003) -> torch.Tensor:
    """
    Jitter points. jittering is per point. 
    Normal distribution is used to jitter points.

    Parameters
    ----------
    pc : torch.Tensor (..., N, 3)
    sigma : float, optional
        Standard deviation of the normal distribution
    clip : float, optional
        Clip the jittered points to be inside the original point cloud
    """
        
    jitter = torch.normal(mean=0, std=sigma, size=pc.shape)
    jitter = torch.clamp(jitter, -1*clip, clip)
    pc = pc + jitter
    
    return pc

def svd_estimate_tf(static, moving, weights=None):
    """
    Estimate the transformation matrix between  two matched point clouds using SVD

    Parameters
    ----------
    static : torch.Tensor # (B, N, 3)
    moving : torch.Tensor # (B, N, 3)
    weights : torch.Tensor # (B, N, N) (optional)
    
    Returns
    -------
    tf : torch.Tensor # (B, 4, 4)
    """
    BS = static.shape[0]
    if weights is None:
        weights = batch_eye(static.shape[:-1], device=static.device, dtype=static.dtype)
    
    static = static.permute(0, 2, 1) # (B, 3, N)
    moving = moving.permute(0, 2, 1) # (B, 3, N)
    
    
    w_i = weights.sum(dim=1, keepdim=True) # (B, 1, N)
    static_mean = torch.sum(static*w_i, dim=2, keepdim=True) / torch.sum(w_i, dim=2, keepdim=True) # (B, 3, 1)
    moving_mean = torch.sum(moving*w_i, dim=2, keepdim=True) / torch.sum(w_i, dim=2, keepdim=True) # (B, 3, 1)
        
    static_ = static - static_mean # (B, 3, N)
    moving_ = moving - moving_mean # (B, 3, N)

    moving_ = torch.bmm(moving_, weights) # (B, 3, N) * (B, N, N) -> (B, 3, N)
    W = torch.bmm(moving_, static_.transpose(2, 1)) # (B, 3, N) * (B, N, 3) -> (B, 3, 3)

    U, _, V = torch.svd(W) # (B, 3, 3)

    reflect = torch.eye(3, device=static.device, dtype=static.dtype).unsqueeze(0).repeat(BS, 1, 1) # (B, 3, 3)
    reflect[:, 2, 2] = torch.det(torch.bmm(V, U.transpose(2, 1))) # (B, 3, 3)
    
    rot = torch.bmm(torch.bmm(V, reflect), U.transpose(2, 1)) # (B, 3, 3) * (B, 3, 3) * (B, 3, 3) -> (B, 3, 3)
    trans = static_mean - torch.bmm(rot, moving_mean) # (B, 3, 1) - (B, 3, 3) * (B, 3, 1) -> (B, 3, 1)
    
    tf = torch.eye(4, device=static.device, dtype=static.dtype).unsqueeze(0).repeat(BS, 1, 1)
    tf[:, :3, :3] = rot
    tf[:, :3, 3] = trans.squeeze(-1)
    
    return tf

def point_to_plane_icp_step(static_pc: torch.Tensor,
                            moving_pc: torch.Tensor,
                            normal_st: torch.Tensor,
                            normal_mv: torch.Tensor = None,
                            weights: torch.Tensor = None):
    """ Perform one step of ICP using point-to-plane error. It is assumed that the point clouds are already matched.

    Parameters
    ----------
    static_pc : torch.Tensor (B, N, 3)
    moving_pc : torch.Tensor (B, N, 3)
    static_normal : torch.Tensor (B, N, 3)
    weights : torch.Tensor (B, N, N) (optional)
    
    Returns
    -------
    tf : torch.Tensor (B, 4, 4)
    """
    if normal_mv is None:
        normal = normal_st
    else:
        normal = (normal_st + normal_mv)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        
    if weights is not None:
        w = weights.sum(dim=2, keepdim=True) # (B, N, 1)
        normal = normal * w.repeat(1, 1, 3) # (B, N, 3)
    
    g_1 = skew(moving_pc).permute(0, 1, 3, 2) # (B, N, 3, 3)
    g_2 = eye_like(g_1) # (B, N, 3, 3)
    g = torch.cat([g_1, g_2], dim=-1) # (B, N, 3, 6)
    
    C = torch.einsum("bndp, bnp -> bnd", g.permute(0, 1, 3, 2), normal) # (B, N, 6, 3) * (B, N, 3) -> (B, N, 6)
    
    A = torch.einsum("bnd, bnp -> bndp", C, C) # (B, N, 6) * (B, N, 6) -> (B, N, 6, 6)
    delta_p = moving_pc - static_pc # (B, N, 3)
    B = torch.einsum("bnd, bnp -> bndp", C, normal) # (B, N, 6) * (B, N, 3) -> (B, N, 6, 3)
    B = torch.einsum("bndp, bnp -> bnd", B, delta_p) # (B, N, 6, 3) * (B, N, 3) -> (B, N, 6)
        
    A = A.sum(dim=1) # (B, 6, 6)
    B = B.sum(dim=1) # (B, 6)
    
    x = torch.linalg.solve(A, B) # (B, 6)
    r = x[:, :3] # (B, 3) (Alpha, Beta, Gamma)
    t = x[:, 3:] # (B, 3)
    
    rot = rotation_from_vector(r) # (B, 4, 4)
    tf = rot.clone()
    tf[:, :3, 3] = t

    tf = inverse(tf)
    return tf