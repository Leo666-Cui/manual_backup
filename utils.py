import os
import random
import numpy as np
import torch
import torch.nn.functional as F

def orthogonal_loss(features: torch.Tensor) -> torch.Tensor:
    """
    计算正交损失。
    鼓励在一个组内（这里是Q个问题）的特征向量相互正交。
    这是通过惩罚其点积矩阵与单位矩阵的差异来实现的。

    参数: features (torch.Tensor): 输入的特征张量，预期维度为 (B, S, Q, D)。
    返回: torch.Tensor: 计算出的标量损失值。
    """
    # 1. 获取维度信息
    B, S, Q, D = features.shape
    device = features.device
    
    # 2. L2归一化: 确保每个特征向量的长度为1 (Norm=1)
    #    这是为了让 F·F^T 的对角线元素趋近于1
    features = F.normalize(features, p=2, dim=-1)
    
    # 3. 向量化处理：将Batch和Slice维度合并，以便使用批处理矩阵乘法
    #    (B, S, Q, D) -> (B * S, Q, D)
    features_reshaped = features.view(B * S, Q, D)

    # 4. 计算 F * F^T 的点积矩阵
    #    输入1: (B*S, Q, D)
    #    输入2: (B*S, Q, D) -> transpose(1, 2) -> (B*S, D, Q)
    #    输出: (B*S, Q, Q)
    #    torch.bmm 是批处理矩阵乘法，会高效地执行 B*S 次独立的 (Q,D) @ (D,Q) 矩阵乘法
    dot_product_matrix = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))

    # 5. 创建我们的目标矩阵：一批单位矩阵；torch.eye(Q) 创建一个 QxQ 的单位矩阵
    identity_matrix = torch.eye(Q, device=device)
    #    将其扩展到与 dot_product_matrix 相同的批次大小
    identity_matrix_batch = identity_matrix.unsqueeze(0).expand(B * S, Q, Q)

    # 6. 计算损失：我们使用均方误差损失 (MSE) 来惩罚点积矩阵与单位矩阵的差异
    #    这会同时促使: 对角线元素 -> 1 (每个向量与自身的点积为1，因为已归一化)，非对角线元素 -> 0 (不同向量之间的点积为0，即正交)
    loss = F.mse_loss(dot_product_matrix, identity_matrix_batch)
    
    return loss


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id, seed=42):
    """Initialize workers with different seeds"""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# Define radiological feature prompts
# ['feat44', 'feat53', 'feat59', 'feat60', 'feat62', 'feat65', 'feat66', 'feat68', 'feat70']
Prompts = [
    # 1. 强化的包膜
    ['Enhancing capsule is absent',
     'Enhancing capsule is present'], 

    # 2. 瘤周异常灌注
    ['Peritumoral abnormal perfusion is absent',
     'Peritumoral abnormal perfusion is present'],

    # 3. 低密度晕
    ['Hypodense halo is absent',
     'Hypodense halo is present (complete or incomplete)'],

    # 4. 冠状强化
    ['Corona enhancement is absent',
     'Corona enhancement is present'],

    # 5. TTPVI
    ['TTPVI is negative',
     'TTPVI is positive'],

    # 6. fade
    ['Fading pattern is absent',
     'Fading pattern is present'],

    # 7. 结中结模式
    ['Nodule-in-nodule architecture is absent',
     'Nodule-in-nodule architecture is present'],

    # 8. 瘤周 wash out
    ['Peritumoral washout is absent',
     'Peritumoral washout is present'],

    # 9. 延迟性中心强化
    ['Delayed central enhancement is absent',
     'Delayed central enhancement is present']
]



rad_feat_ind = [44, 53, 59, 60, 62, 65, 66, 68, 70]
# rad_feat_ind = [4, 9, 11, 15, 20, 25, 27, 29, 30, 32, 36]
rad_feat_ind = ['feat'+str(i) for i in rad_feat_ind]
# List of samples to exclude from analysis
drop_id = [
    1835136,
    1068358,
    1684755,
    1749849,
]

# Default session type
session = 'ap' 
