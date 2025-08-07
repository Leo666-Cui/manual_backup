import SimpleITK as sitk
import torch
import torch.nn.functional as F
import numpy as np
import imageio # 用于保存图像
import random
import os
from scipy import ndimage # 确保导入scipy
from tqdm import tqdm
import shutil 


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
    # print(f"slice_nonzero_sums: {slice_nonzero_sums}")
    
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
        # print(f"indices_of_valid_indices: {indices_of_valid_indices}")
        selected_indices = valid_indices[indices_of_valid_indices]
    else:
        # 如果有效切片不足10张
        print(f"有效切片不足 {num_slices_to_select} 张，将重复最后一个有效切片进行填充。")
        selected_indices = valid_indices.tolist() # 转为列表以方便追加
        # 获取最后一个有效切片的真实索引
        last_valid_index = valid_indices[-1]
        # 计算还需要补充多少张
        num_to_pad = num_slices_to_select - len(valid_indices)
        # 使用最后一个【有效】的索引 (last_valid_index) 来进行填充
        padding_indices = [last_valid_index] * num_to_pad
        selected_indices.extend(padding_indices)
        selected_indices = np.array(selected_indices) 
        
    
    print(f"最终选出的10个索引: {selected_indices}")
    return volume[selected_indices], selected_indices

def crop_to_roi(image_slice, padding=10):
    """
    将单张2D图像裁剪到其感兴趣区域 (ROI)。
    ROI被定义为图像中所有非零像素的最小边界框。
    """
    # 找到所有非零像素的坐标
    # np.where 会返回一个元组，第一个元素是所有非零像素的行索引，第二个是列索引
    coords = np.where(image_slice > 0)
    
    # 如果图像是全黑的，coords 会是空的
    if len(coords[0]) == 0:
        # 对于全黑的图像，我们可以直接返回一个很小的黑色方块，或者原图
        return image_slice # 或者 return np.zeros((10, 10), dtype=np.uint8)

    # 计算边界框的四个角点
    min_row, max_row = np.min(coords[0]), np.max(coords[0])
    min_col, max_col = np.min(coords[1]), np.max(coords[1])
    
    # 2. 在原始边界框的基础上，向外扩展'padding'个像素
    #    使用 np.maximum 确保坐标不会小于0
    min_row_padded = np.maximum(0, min_row - padding)
    min_col_padded = np.maximum(0, min_col - padding)
    
    #    使用 np.minimum 确保坐标不会超过图像的原始尺寸
    max_row_padded = np.minimum(image_slice.shape[0], max_row + padding)
    max_col_padded = np.minimum(image_slice.shape[1], max_col + padding)

    # 使用切片操作进行裁剪 (+1 是因为切片不包含结束索引)
    cropped_image = image_slice[min_row_padded : max_row_padded, min_col_padded : max_col_padded]
    
    return cropped_image

def create_montage(image_slices, grid_dims, output_path):
    """
    将一系列【尺寸不一】的2D图像切片拼接成一张大的网格图。
    它会自动将所有小图填充到最大尺寸，以保持对齐。
    """
    rows, cols = grid_dims
    if len(image_slices) > rows * cols:
        raise ValueError("Grid dimensions are too small for the number of slices.")

    # 1. 找到所有裁剪后图像中的最大高度和最大宽度
    max_h = max(img.shape[0] for img in image_slices)
    max_w = max(img.shape[1] for img in image_slices)
    print(f"所有裁剪后图像的最大尺寸为: ({max_h}, {max_w})")

    # 2. 创建一个大的画布
    montage_h = rows * max_h
    montage_w = cols * max_w
    montage = np.zeros((montage_h, montage_w), dtype=np.uint8)

    # 3. 将每个切片粘贴到画布上（居中粘贴）
    for i, slice_img in enumerate(image_slices):
        row_idx = i // cols
        col_idx = i % cols
        
        # 计算粘贴的起始位置，实现居中效果
        y_start = (row_idx * max_h) + (max_h - slice_img.shape[0]) // 2
        y_end = y_start + slice_img.shape[0]
        x_start = (col_idx * max_w) + (max_w - slice_img.shape[1]) // 2
        x_end = x_start + slice_img.shape[1]
        
        montage[y_start:y_end, x_start:x_end] = slice_img

    # 保存蒙太奇图像
    imageio.imwrite(output_path, montage)
    print(f"尺寸不一的蒙太奇图像已保存到 {output_path}")


# --- 主流程  ---

# 1. 定义基础的文件路径
base_nii_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/test_Internal_nii2'
base_roi_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/test_Internal_roi'
# 这是存放50张图的新基础路径
base_50_slices_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/50_slices_image'
# 这是存放最终20张裁剪图的新基础路径
base_cropped_output_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/cropped_20_slices_image'
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

    # 我们先在这里创建，并清理旧文件
    patient_cropped_output_folder = os.path.join(base_cropped_output_path, patient_id)
    # 所有子文件都会被删除
    if os.path.exists(patient_cropped_output_folder):
        shutil.rmtree(patient_cropped_output_folder)
    os.makedirs(patient_cropped_output_folder)

    # 循环处理 ap 和 pvp 两个期相
    for phase in ['ap', 'pvp']:
        print(f"\n  -- 正在处理 {phase.upper()} 期相 --")

        # 3.1 构建当前病人和期相的输入输出路径
        nii_gz_path = os.path.join(base_nii_path, patient_id, f'{phase}.nii.gz')
        nrrd_path = os.path.join(base_roi_path, patient_id, f'{phase}.nrrd')
        
        # 为50张图创建期相专属的子文件夹
        output_50_slices_folder = os.path.join(base_50_slices_path, patient_id, phase)

        # 3.2 检查输入文件是否存在
        if not os.path.exists(nii_gz_path) or not os.path.exists(nrrd_path):
            print(f"  警告: 找不到病人 {patient_id} 的 {phase.upper()} 文件，跳过该期相。")
            continue # 跳到下一个病人

        # 3.3 创建当前病人的50张图输出文件夹
        os.makedirs(output_50_slices_folder, exist_ok=True)
        
        # 3.4 加载NIfTI文件
        # nii.gz -> SimpleITK.Image -> NumPy
        try:
            image_volume_array = sitk.GetArrayFromImage(sitk.ReadImage(nii_gz_path))
            seg_volume_array = sitk.GetArrayFromImage(sitk.ReadImage(nrrd_path))
        except Exception as e:
            print(f"  加载病人 {patient_id} 的 {phase.upper()} 文件时出错: {e}，跳过该期相。")
            continue

        # 3.5 将3D数据块的深度标准化为50个切片
        padded_img_volume, padded_seg_volume = pad_to_max_slices(image_volume_array, seg_volume_array, max_slices=max_slices_to_save)
        
        # 3.6 对整个50切片的数据块应用窗宽窗位调整
        windowed_volume = apply_window(padded_img_volume)

        # 3.7 对掩码进行膨胀以包含边缘
        binary_seg_mask = (padded_seg_volume > 0).astype(np.uint8)
        dilated_mask = ndimage.binary_dilation(binary_seg_mask, iterations=10).astype(np.uint8)
        
        # 3.8 应用膨胀后的掩码
        masked_windowed_volume = windowed_volume * dilated_mask
        # masked_windowed_volume = windowed_volume

        # 3.9 循环遍历处理好的50个切片，并逐一保存
        for i in range(masked_windowed_volume.shape[0]):
            slice_to_save = masked_windowed_volume[i, :, :]
            output_path = os.path.join(output_50_slices_folder, f'slice_{str(i).zfill(2)}.png')
            imageio.imwrite(output_path, slice_to_save)
        print(f"  病人 {patient_id} 的 {phase.upper()} 期相的50张切片已保存。")

        # 在50张CT图等间距挑选10张有效的
        ten_processed_slices, selected_indices = select_slices_valid_range_linspace(masked_windowed_volume)
        # montage_file_path = "/home/yxcui/FM-Bridge/testing_file/test_dataset/Montage"
        # output_montage_path = os.path.join(montage_file_path, f'{patient_id}_montage.png')
        # create_montage(ten_processed_slices, grid_dims=(2, 5), output_path=output_montage_path)
        # print(f"病人 {patient_id} 的蒙太奇图像已成功保存到 '{output_montage_path}'")

        # cropped 10 images from ten_processed_slices
        phase_specific_folder = os.path.join(base_cropped_output_path, patient_id, phase)
        # 创建一个phase的文件夹
        os.makedirs(phase_specific_folder)
        print(f"  正在对选出的 {len(ten_processed_slices)} 张 {phase.upper()} 切片进行裁剪并保存...")

        # 我们使用 zip 将切片和它们的原始索引用_来打包遍历
        for original_idx, slice_to_crop in zip(selected_indices, ten_processed_slices):
            # 裁剪
            cropped_slice = crop_to_roi(slice_to_crop)
            # 使用原始索引来命名文件，例如 cropped_slice_25.png
            output_path = os.path.join(phase_specific_folder, f'{phase}_cropped_{str(original_idx).zfill(2)}.png')
            # 保存
            imageio.imwrite(output_path, cropped_slice)

        print(f"病人 {patient_id} 的 {len(ten_processed_slices)} 张 {phase.upper()} 裁剪切片已保存。")

print("\n\n所有病人处理完毕!")

