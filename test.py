import SimpleITK as sitk
import torch
import torch.nn.functional as F
import numpy as np
import imageio # 用于保存图像
import random
import os
from scipy import ndimage # 确保导入scipy
from tqdm import tqdm
import pandas as pd


# label_df = pd.read_csv('/home/yxcui/FM-Bridge/Dataset/label.csv')
# rad_feat_df = pd.read_csv('/home/yxcui/FM-Bridge/Dataset/rad_cli_feat.csv')
# label_df = label_df.merge(rad_feat_df, on=['pathological number', 'label'], how='left')
# label_df.to_csv('label_df.csv', index=False)

# print("已成功将完整的合并数据保存到 label_df.csv 文件中。")

# 测试每个病人Ct图里的50张CT每一张的像素点数量
def pad_to_max_slices(img_volume, seg_volume, max_slices=50):
    """将3D图像和分割图填充或裁剪到指定的切片数。
    为了使用 torch.nn.functional.pad 这个高效的填充功能,代码首先将输入的NumPy数组转换成了一个PyTorch张量
    return 之前,它又将处理好的PyTorch张量通过 .numpy() 方法转换回了NumPy数组"""
    # 将 NumPy 数组转换为 PyTorch 张量以使用 F.pad
    img_tensor = torch.from_numpy(img_volume.astype(np.float32))
    seg_tensor = torch.from_numpy(seg_volume.astype(np.float32))

    # PyTorch 的 pad 函数作用于最后的维度，而我们的切片维度在最前面 (z, y, x)
    # 所以我们需要先调整维度顺序 (z, y, x) -> (y, x, z)
    img_tensor = img_tensor.permute(1, 2, 0)
    seg_tensor = seg_tensor.permute(1, 2, 0)
    
    current_slices = img_tensor.shape[-1]
    
    if current_slices > max_slices:
        start = (current_slices - max_slices) // 2
        img_tensor = img_tensor[..., start:start + max_slices]
        seg_tensor = seg_tensor[..., start:start + max_slices]
    elif current_slices < max_slices:
        pad_size = max_slices - current_slices
        # (左, 右) -> (0, pad_size) 表示只在右边（末尾）填充
        img_tensor = F.pad(img_tensor, (0, pad_size), mode='constant', value=0)
        seg_tensor = F.pad(seg_tensor, (0, pad_size), mode='constant', value=0)

    # 将维度顺序恢复为 (z, y, x)
    img_processed = img_tensor.permute(2, 0, 1)
    seg_processed = seg_tensor.permute(2, 0, 1)

    # 将张量转回 NumPy 数组
    return img_processed.numpy(), seg_processed.numpy()

def apply_window(img, window_center=50, window_width=100): # 得到3D的NumPy数组
    """对 NumPy 数组应用窗宽窗位调整，内部使用 PyTorch 进行计算。"""
    # 将输入的 NumPy 数组转换为 PyTorch 张量
    img_tensor = torch.from_numpy(img.astype(np.float32))

    window_center = window_center + random.uniform(-20, 20)
    window_width = window_width * random.uniform(0.8, 1.2)

    window_min = window_center - window_width/2
    window_max = window_center + window_width/2
    
    # 在 PyTorch 张量上执行所有数学运算
    img_tensor = (img_tensor - window_min) / (window_max - window_min)
    img_tensor = torch.clip(img_tensor, 0, 1)* 255.0
    img_rounded_tensor = torch.round(img_tensor)

    # 将最终的张量转换回 NumPy 数组，并设定正确的数据类型
    img_processed = img_rounded_tensor.cpu().numpy().astype(np.uint8)

    return img_processed

# 1. 定义基础的文件路径
# 1042520 1047386 1049029 1050496 1058059 1059537 1068358 1096907 1120387 1120915
nii_gz_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/test_Internal_nii2/1120915/ap.nii.gz'
nrrd_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/test_Internal_roi/1120915/ap.nrrd'
max_slices_to_save = 50

try:
    sitk_image = sitk.ReadImage(nii_gz_path)
    sitk_seg = sitk.ReadImage(nrrd_path)
    image_volume_array = sitk.GetArrayFromImage(sitk_image) # 图像转换为NumPy数组
    seg_volume_array = sitk.GetArrayFromImage(sitk_seg) 
except Exception as e:
    print(f"加载文件时出错: {e}")
    exit()


# 3.5 将3D数据块的深度标准化为50个切片
padded_img_volume, padded_seg_volume = pad_to_max_slices(image_volume_array, seg_volume_array, max_slices=max_slices_to_save)

# 3.6 对整个50切片的数据块应用窗宽窗位调整
windowed_volume = apply_window(padded_img_volume)

# 3.7 对掩码进行膨胀以包含边缘
binary_seg_mask = (padded_seg_volume > 0).astype(np.uint8)
dilated_mask = ndimage.binary_dilation(binary_seg_mask, iterations=5).astype(np.uint8)

# 3.8 应用膨胀后的掩码
masked_windowed_volume = windowed_volume * dilated_mask

#第38张图
# for i in range(max_slices_to_save):
#     target_slice = masked_windowed_volume[i, :, :]
#     non_masked_pixel_count = np.count_nonzero(target_slice)

#     print(f'第{i}张图像素总和是:{non_masked_pixel_count}')

for i in range(max_slices_to_save):
    target_slice = windowed_volume[i, :, :]
    non_masked_pixel_count = np.count_nonzero(target_slice)

    print(f'第{i}张图像素总和是:{non_masked_pixel_count}')
