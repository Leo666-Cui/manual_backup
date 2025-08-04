import SimpleITK as sitk
import torch
import torch.nn.functional as F
import numpy as np
import imageio # 用于保存图像
import random
import os
from scipy import ndimage # 确保导入scipy
from tqdm import tqdm


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

def select_slices_valid_range_linspace(volume, num_slices_to_select=10,threshold=300):
    """
    在有效切片范围内进行等间距采样。如果有效切片不足10张,则重复最后一个有效切片以补足10张。
    """
    # 1. 检测每张图里有没有信息 (计算每个切片的像素总和)
    slice_nonzero_sums = np.sum(volume, axis=(1, 2))
    print(f"slice_nonzero_sums: {slice_nonzero_sums}")
    
    # 2. 找到所有非黑切片的索引
    # valid_indices = np.where(slice_nonzero_sums > threshold)[0]
    valid_indices = np.where((slice_nonzero_sums > threshold) & (slice_nonzero_sums != 262144))[0]
    print(f"valid_indices: {valid_indices}")
    
    if len(valid_indices) == 0:
        print("警告: 未找到任何有效切片，返回空列表。")
        return np.array([])
    print(f"检测到 {len(valid_indices)} 张有效切片 (像素和 > {threshold})。")
    
    # 3. 确定有效范围的开始和结束
    start_index = valid_indices[0]
    end_index = valid_indices[-1]
    
    # 4. 根据有效切片数量选择采样策略
    if len(valid_indices) >= num_slices_to_select:
        # 先创建一组针对 valid_indices 列表的索引，使用这组新索引，从 valid_indices 中挑选出最终的切片索引
        indices_of_valid_indices = np.linspace(0, len(valid_indices) - 1, num=num_slices_to_select, dtype=int)
        print(f"indices_of_valid_indices: {indices_of_valid_indices}")
        selected_indices = valid_indices[indices_of_valid_indices]
    else:
        # 如果有效切片不足10张
        print(f"有效切片不足 {num_slices_to_select} 张，将重复最后一个有效切片进行填充。")
        # 首先，全选所有已有的有效切片
        selected_indices = valid_indices
        # 计算还需要补充多少张
        num_to_pad = num_slices_to_select - len(valid_indices)
        # 创建一个包含多个“最后一个有效切片索引”的列表
        padding_indices = [end_index + 1] * num_to_pad
        # 将补充的索引追加到已选索引的末尾
        selected_indices = np.concatenate([selected_indices, padding_indices])
        
    
    print(f"最终选出的10个索引: {selected_indices}")
    return volume[selected_indices]

def create_montage(image_slices, grid_dims, output_path):
    """
    将一系列2D图像切片拼接成一张大的网格图。

    :param image_slices: 一个包含多个2D NumPy数组 (uint8) 的列表。
    :param grid_dims: 一个元组(rows, cols)，定义网格的尺寸。
    :param output_path: 保存蒙太奇图像的路径。
    """
    rows, cols = grid_dims
    num_slices = len(image_slices)
    if num_slices > rows * cols:
        raise ValueError("Grid dimensions are too small for the number of slices.")

    # 获取单个切片的尺寸
    slice_h, slice_w = image_slices[0].shape
    
    # 创建一个大的画布
    montage_h = rows * slice_h
    montage_w = cols * slice_w
    montage = np.zeros((montage_h, montage_w), dtype=np.uint8)

    # 将每个切片粘贴到画布上
    for i, slice_img in enumerate(image_slices):
        row_idx = i // cols
        col_idx = i % cols
        
        y_start = row_idx * slice_h
        y_end = y_start + slice_h
        x_start = col_idx * slice_w
        x_end = x_start + slice_w
        
        montage[y_start:y_end, x_start:x_end] = slice_img

    # 保存蒙太奇图像
    imageio.imwrite(output_path, montage)



# --- 主流程  ---

# 1. 定义基础的文件路径
base_nii_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/test_Internal_nii2'
base_roi_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/test_Internal_roi'
base_output_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/50_slices_image'
max_slices_to_save = 50

# 2. 自动获取所有病人的ID列表
#    我们通过扫描nii文件夹下的所有子文件夹来获取病人ID
try:
    patient_ids = [d for d in os.listdir(base_nii_path) if os.path.isdir(os.path.join(base_nii_path, d))]
    print(f"找到了 {len(patient_ids)} 个病人: {patient_ids}")
except FileNotFoundError:
    print(f"错误：找不到输入文件夹 {base_nii_path}。请检查路径是否正确。")
    exit()

# 3. 循环处理每一个病人
for patient_id in tqdm(patient_ids, desc="Processing Patients"):
    print(f"\n--- 开始处理病人: {patient_id} ---")

    # 3.1 构建当前病人的输入和输出路径
    nii_gz_path = os.path.join(base_nii_path, patient_id, 'ap.nii.gz')
    nrrd_path = os.path.join(base_roi_path, patient_id, 'ap.nrrd')
    output_folder = os.path.join(base_output_path, patient_id)

    # 3.2 检查输入文件是否存在
    if not os.path.exists(nii_gz_path) or not os.path.exists(nrrd_path):
        print(f"警告: 找不到病人 {patient_id} 的完整文件，跳过该病人。")
        continue # 跳到下一个病人

    # 3.3 创建当前病人的输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 3.4 加载NIfTI文件
    try:
        sitk_image = sitk.ReadImage(nii_gz_path)
        sitk_seg = sitk.ReadImage(nrrd_path)
        image_volume_array = sitk.GetArrayFromImage(sitk_image)
        seg_volume_array = sitk.GetArrayFromImage(sitk_seg)
    except Exception as e:
        print(f"加载病人 {patient_id} 的文件时出错: {e}，跳过该病人。")
        continue

    # 3.5 将3D数据块的深度标准化为50个切片
    padded_img_volume, padded_seg_volume = pad_to_max_slices(image_volume_array, seg_volume_array, max_slices=max_slices_to_save)
    
    # 3.6 对整个50切片的数据块应用窗宽窗位调整
    windowed_volume = apply_window(padded_img_volume)

    # 3.7 对掩码进行膨胀以包含边缘
    binary_seg_mask = (padded_seg_volume > 0).astype(np.uint8)
    dilated_mask = ndimage.binary_dilation(binary_seg_mask, iterations=5).astype(np.uint8)
    
    # 3.8 应用膨胀后的掩码
    # masked_windowed_volume = windowed_volume * dilated_mask
    masked_windowed_volume = windowed_volume

    # 3.9 循环遍历处理好的50个切片，并逐一保存
    for i in range(masked_windowed_volume.shape[0]):
        slice_to_save = masked_windowed_volume[i, :, :]
        output_path = os.path.join(output_folder, f'slice_{str(i).zfill(2)}.png')
        imageio.imwrite(output_path, slice_to_save)

    print(f"病人 {patient_id} 的 {max_slices_to_save} 张切片已成功保存到 '{output_folder}'")

    # 在50张CT图等间距挑选10张有效的
    ten_processed_slices = select_slices_valid_range_linspace(masked_windowed_volume)
    montage_file_path = "/home/yxcui/FM-Bridge/testing_file/test_dataset/Montage"
    output_montage_path = os.path.join(montage_file_path, f'{patient_id}_montage.png')
    create_montage(ten_processed_slices, grid_dims=(2, 5), output_path=output_montage_path)
    print(f"病人 {patient_id} 的蒙太奇图像已成功保存到 '{output_montage_path}'")

print("\n\n所有病人处理完毕+拼接!")



