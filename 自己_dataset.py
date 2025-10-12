import os
import random
import numpy as np
import torch
import monai
from monai.data import ImageReader
from torch.utils.data import Dataset as TorchDataset
from monai.transforms import apply_transform, Randomizable, Resize, LoadImage
import torchvision.transforms.functional as TF
import math  
from scipy import ndimage  
import torch.nn.functional as F
import torchvision.transforms as transforms

# 确保您的MONAI版本支持这些功能
# print(f"MONAI Version: {monai.__version__}")

class PatientMultiSliceDataset(TorchDataset):
    """
    一个用于处理2D PNG图像和掩码的数据集类。
    - 继承自 monai.data.ImageDataset，利用其加载器。
    - 同步图像和掩码的随机变换。
    - 结合了MONAI和Torchvision的变换流程。
    """
    def __init__(self, image_files, seg_files, labels, rad_feat, transform=None, seg_transform=None, train=False, base_dilation_iterations=10, random_dilation_range=5):
        super().__init__()
        
        # 存储额外的文件和参数
        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.rad_feat = rad_feat
        self.transform = transform
        self.seg_transform = seg_transform
        self.train = train
        self.base_dilation_iterations = base_dilation_iterations
        self.random_dilation_range = random_dilation_range

        # 用于同步随机变换的随机数生成器
        self.rng = np.random.RandomState()
        # 实例化一个标准的LoadImage加载器。
        # image_only=True 表示我们只关心图像数据，不关心元数据（比如仿射矩阵等）
        self.loader = LoadImage(image_only=True)

        self.slice_transforms = transforms.Compose([
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=(-10, 10),
                fill=0
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]) if train else None

    def _pad_to_diagonal_square(self, img, fill=0):
        """
        [内部方法] 将PIL图像补白到一个以其对角线为边长的正方形。
        """
        w, h = img.size
        diagonal = math.ceil(math.sqrt(w**2 + h**2))
        pad_h = diagonal - h
        pad_w = diagonal - w
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - (pad_w // 2),
            pad_h - (pad_h // 2),
        )
        return TF.pad(img, padding, fill=fill)

    def _random_dilate_mask_ndimage(self, seg_tensor):
        """
        [内部方法] 使用ndimage对输入的PyTorch Tensor掩码进行随机膨胀。
        """
        # 仅在训练模式下执行
        if not self.train:
            return seg_tensor

        # 1. 计算随机的迭代次数
        offset = random.randint(-self.random_dilation_range, self.random_dilation_range)
        iterations = self.base_dilation_iterations + offset
        
        # 如果迭代次数小于等于0，则不进行膨胀
        if iterations <= 0:
            return seg_tensor
            
        # 2. 将 PyTorch Tensor 转换为 NumPy Array
        #    ndimage 需要CPU上的NumPy数组
        device = seg_tensor.device
        # MONAI加载的seg是(1, H, W)，ndimage需要(H, W)
        seg_np = seg_tensor.squeeze(0).cpu().numpy()

        # 3. 执行ndimage膨胀
        #    ndimage.binary_dilation需要一个布尔类型的输入
        dilated_np = ndimage.binary_dilation((seg_np > 0), iterations=iterations)

        # 4. 将结果转回 PyTorch Tensor
        #    先将布尔型转为浮点型，再增加通道维度，并送回原来的设备
        dilated_tensor = torch.from_numpy(dilated_np.astype(np.float32)).unsqueeze(0).to(device)
        
        return dilated_tensor

    def _crop_based_on_mask_roi(self, img_np, mask_np, padding=10):
        """
        [内部方法] 根据掩码(mask)的ROI，同时裁剪图像和掩码的Numpy数组。
        这是为了确保裁剪后两者尺寸和位置完全对应。
        """
        # 关键：从掩码中找到所有非零像素的坐标
        coords = np.where(mask_np > 0)
        
        if len(coords[0]) == 0:
            return img_np, mask_np # 如果掩码为空，不进行裁剪

        # 计算掩码的边界框
        min_row, max_row = np.min(coords[0]), np.max(coords[0])
        min_col, max_col = np.min(coords[1]), np.max(coords[1])
        
        # 应用padding，并确保坐标不越界
        h, w = mask_np.shape
        min_row_padded = np.maximum(0, min_row - padding)
        min_col_padded = np.maximum(0, min_col - padding)
        max_row_padded = np.minimum(h, max_row + padding + 1)
        max_col_padded = np.minimum(w, max_col + padding + 1)

        # 使用【同一套坐标】裁剪图像和掩码
        cropped_image = img_np[min_row_padded:max_row_padded, min_col_padded:max_col_padded]
        cropped_mask = mask_np[min_row_padded:max_row_padded, min_col_padded:max_col_padded]
        
        return cropped_image, cropped_mask

    def __len__(self):
        """
        返回数据集中样本的总数，即病人的数量。
        """
        return len(self.image_files)

    def __getitem__(self, index):
        # 1. 获取该病人的所有文件路径
        patient_image_paths = self.image_files[index]
        patient_seg_paths = self.seg_files[index]

        # 用于存储处理后的每个2D切片
        processed_img_slices = []
        processed_seg_slices = []

        # 设置一个用于本次样本所有随机操作的同步种子
        seed = self.rng.randint(2**32)

        # 2. 循环处理该病人的每一张切片
        for img_path, seg_path in zip(patient_image_paths, patient_seg_paths):
            # LoadImage的调用方式是直接调用实例本身，而不是调用.read()方法
            # 它直接返回一个PyTorch张量
            img_slice_tensor = self.loader(img_path)
            seg_slice_tensor = self.loader(seg_path)
            
            # 确保是单通道
            if img_slice_tensor.shape[0] > 1: img_slice_tensor = img_slice_tensor[0:1]
            if seg_slice_tensor.shape[0] > 1: seg_slice_tensor = seg_slice_tensor[0:1]
            
            # 2.2 对掩码进行随机膨胀
            seg_slice_tensor = self._random_dilate_mask_ndimage(seg_slice_tensor)

            # 2.3 应用2D数据增强 (如果是训练模式)
            if self.train and self.slice_transforms:
                # 转为PIL Image以使用torchvision
                img_slice_pil = TF.to_pil_image(img_slice_tensor)
                seg_slice_pil = TF.to_pil_image(seg_slice_tensor)
                
                # 使用相同的种子确保变换同步
                random.seed(seed)
                torch.manual_seed(seed)
                img_slice_aug = self.slice_transforms(img_slice_pil)
                
                random.seed(seed)
                torch.manual_seed(seed)
                seg_slice_aug = self.slice_transforms(seg_slice_pil)

                # 变换后转回Tensor
                img_slice_tensor = TF.to_tensor(img_slice_aug)
                seg_slice_tensor = TF.to_tensor(seg_slice_aug)

            # 将处理后的切片添加到列表中
            processed_img_slices.append(img_slice_tensor)
            processed_seg_slices.append(seg_slice_tensor)

        # 3. 将处理后的2D切片列表堆叠成一个3D/4D张量
        # torch.stack会在新的维度上堆叠，这里我们选择dim=-1
        # 结果形状: (C, H, W, S)，其中 S 是切片数 (15)
        final_img = torch.stack(processed_img_slices, dim=-1)
        final_seg = torch.stack(processed_seg_slices, dim=-1)
        
        # 4. 最终的尺寸和格式调整
        # MONAI的Resize可以直接作用于 (C, H, W, S) 的Tensor
        resizer = Resize(spatial_size=(224, 224, 15)) # 确保深度也是15
        final_img = resizer(final_img)
        final_seg = resizer(final_seg)
        # print(f"final_img.shape before: {final_img.shape}")
        # print(f"final_seg.shape before: {final_seg.shape}")
        final_img = final_img.repeat(3, 1, 1, 1) # 第 0 维度 通道维度重复3次,最终通道数从1变为3
        final_seg = final_seg.repeat(3, 1, 1, 1)
        # print(f"final_img.shape after: {final_img.shape}")
        # print(f"final_seg.shape after: {final_seg.shape}")

        # 确保掩码是二值的 (0.0 或 1.0)
        final_seg = (final_seg >= 0.5).float()
        
        # 5. 组合所有数据并返回
        label = self.labels[index]
        rad_feature = self.rad_feat[index]
        patient_id = patient_image_paths[0].split('/')[-1].split('_')[0]

        data = (final_img, final_seg, label, rad_feature, patient_id)
        
        return data
