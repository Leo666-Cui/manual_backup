import os
import random
import numpy as np
import torch
import monai
from monai.data import ImageDataset
from monai.transforms import apply_transform, Randomizable, Resize
import torchvision.transforms.functional as TF

# 确保您的MONAI版本支持这些功能
print(f"MONAI Version: {monai.__version__}")

class Png2dDataset(ImageDataset):
    """
    一个用于处理2D PNG图像和掩码的数据集类。
    - 继承自 monai.data.ImageDataset，利用其加载器。
    - 同步图像和掩码的随机变换。
    - 结合了MONAI和Torchvision的变换流程。
    """
    def __init__(self, image_files, seg_files, labels, rad_feat, 
                 transform=None, seg_transform=None, train=False):

        # 使用MONAI的ImageDataset的加载器 (默认会使用Pillow来加载PNG)
        # image_only=False 会让加载器同时返回图像数据和元数据
        super().__init__(image_files, labels=labels, transform=None, image_only=False)
        
        # 存储额外的文件和参数
        self.seg_files = seg_files
        self.rad_feat = rad_feat
        self.train = train
        
        # 分开存储MONAI和Torchvision的变换
        self.slice_transforms = slice_transforms

        # 用于同步随机变换的随机数生成器
        self.rng = np.random.RandomState()

        self.slice_transforms = transforms.Compose([
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(0.5, 0.5),
                scale=(0.6, 1.4),
                shear=(-10, 10),
                fill=0
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]) if train else None

    def _pad_to_diagonal_square(self, img: Image.Image, fill=0) -> Image.Image:
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

    def _crop_based_on_mask_roi(self, img_np: np.ndarray, mask_np: np.ndarray, padding=10) -> (np.ndarray, np.ndarray):
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


    def __getitem__(self, index: int):
        # 1. 加载数据 & 设置同步种子
        seed = self.rng.randint(2**32)
        
        # 使用父类(ImageDataset)的加载器加载图像和元数据
        img, meta_data = self.loader(self.image_files[index])
        
        seg, seg_meta_data = None, None
        if self.seg_files is not None:
            seg, seg_meta_data = self.loader(self.seg_files[index])

        # 2. 应用MONAI的变换 (通常用于加载、增加通道、调整方向等)
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=seed)
            img, meta_data = apply_transform(self.transform, (img, meta_data), map_items=False, unpack_items=True)

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=seed)
            seg, seg_meta_data = apply_transform(self.seg_transform, (seg, seg_meta_data), map_items=False, unpack_items=True)
            
        # 3. 应用基于Torchvision的2D切片变换 (通常用于数据增强)
        #    这是适配后的核心逻辑，直接对2D Tensor操作
        if self.train is not None:
            # MONAI加载的图像是Tensor，需要转为PIL Image以使用torchvision
            # 假设图像是 [C, H, W]，我们取第一个通道
            img_slice_pil = TF.to_pil_image(img[0])
            seg_slice_pil = TF.to_pil_image(seg[0])
            
            # pad到正方形，再使用变换
            img_padded = self._pad_to_diagonal_square(img_slice_pil)
            seg_padded = self._pad_to_diagonal_square(seg_slice_pil)

            # 使用相同的种子确保变换同步
            random.seed(seed)
            torch.manual_seed(seed)
            img_slice_aug = self.slice_transforms(img_padded)
            
            random.seed(seed)
            torch.manual_seed(seed)
            seg_slice_aug = self.slice_transforms(seg_padded)

            # 变换后转回Tensor
            img = TF.to_tensor(img_slice_aug)
            seg = TF.to_tensor(seg_slice_aug)


        # 4. 最终尺寸调整 (Resize)
        # MONAI的Resize可以直接作用于 [C, H, W] 的Tensor
        resizer = Resize(spatial_size=(224, 224))
        img_final = resizer(img_final)
        seg_final = resizer(seg_final)

        # 确保掩码是二值的 (0.0 或 1.0)
        if seg is not None:
            seg = (seg >= 0.5).float()

        # 5. 组合所有数据并返回
        label = self.labels[index]
        rad_feature = self.rad_feat[index]
        
        # 从文件名提取ID
        patient_id = self.image_files[index].split('/')[-1].split('_')[0]

        # 按照您的要求构建返回的元组
        data = (img_final, seg_final, label, rad_feature, patient_id)
        
        return data




# -------------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose
import torchvision.transforms as T

# --- 1. 准备文件列表和元数据 (假设这些已准备好) ---
# 您可以运行之前的脚本来生成这些文件列表
img_dir = '/home/yxcui/FM-Bridge/testing_file/test_dataset/img_encoder_nii'
roi_dir = '/home/yxcui/FM-Bridge/testing_file/test_dataset/img_encoder_roi'

# os.listdir返回的文件名可能顺序不一致，排序可以确保img和seg一一对应
train_images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
train_segs = sorted([os.path.join(roi_dir, f) for f in os.listdir(roi_dir)])

num_samples = len(train_images)
# 找到所有病人ID，找到他们的label
train_labels = torch.randint(0, 2, (num_samples, 9)).float() 
train_rad_feat = torch.rand(num_samples, 9)

# --- 2. 定义变换 ---
train_transforms = Compose([EnsureChannelFirst()])
val_transforms = Compose([EnsureChannelFirst()])

# --- 3. 实例化Dataset和DataLoader ---
print(f"准备创建含有 {num_samples} 个样本的数据集...")
train_ds = Png2dDataset(
    image_files=train_images,
    seg_files=train_segs,
    labels=train_labels,
    rad_feat=train_rad_feat,
    transform=train_transforms,
    seg_transform=train_transforms,
    train=True
)

# shuffle要和文本的顺序一样
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn)

# --- 4. 测试DataLoader ---
print("从DataLoader中取出一批数据进行测试...")
batch_data = next(iter(train_loader))
batch_img, batch_seg, batch_label, batch_rad, batch_id = batch_data

print(f"图像批次维度: {batch_img.shape}")      # 应该为 torch.Size([8, 1, 224, 224])
print(f"掩码批次维度: {batch_seg.shape}")      # 应该为 torch.Size([8, 1, 224, 224])
print(f"标签批次维度: {batch_label.shape}")    # 应该为 torch.Size([8, 9])
print(f"特征批次维度: {batch_rad.shape}")    # 应该为 torch.Size([8, 9])
print(f"批次中的ID示例: {batch_id[0]}")      # 打印第一个样本的ID
