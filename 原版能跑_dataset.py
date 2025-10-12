import torch
import numpy as np
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from monai.data import ImageDataset
from monai.transforms import Randomizable, apply_transform
import monai.transforms

class Dataset(ImageDataset):
    def __init__(self, image_files, seg_files, labels, rad_feat, transform=None, seg_transform=None, train=None):
        super().__init__(image_files, seg_files, labels, transform=transform, seg_transform=seg_transform)

        self.base_window_center = 50
        self.base_window_width = 100
        self.train = train
        self.rad_feat = rad_feat
        self.max_slices = 50  # Maximum number of slices to pad to

        self.rng = np.random.RandomState(42)

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

    def pad_to_max_slices(self, img, seg):
        """Pad or crop the image and segmentation to have max_slices in the last dimension."""
        current_slices = img.shape[-1]
        
        # Create a mask of ones with current_slices length
        valid_mask = torch.ones(current_slices)
        
        if current_slices > self.max_slices:
            # If we have more slices than max_slices, take center slices
            start = (current_slices - self.max_slices) // 2
            img = img[..., start:start + self.max_slices]
            seg = seg[..., start:start + self.max_slices]
            valid_mask = valid_mask[start:start + self.max_slices]
        elif current_slices < self.max_slices:
            # If we have fewer slices than max_slices, pad with zeros only on the right
            pad_size = self.max_slices - current_slices
            img = F.pad(img, (0, pad_size), mode='constant', value=0)
            seg = F.pad(seg, (0, pad_size), mode='constant', value=0)
            # Pad the mask with zeros
            valid_mask = F.pad(valid_mask, (0, pad_size), mode='constant', value=0)
            
        return img, seg, valid_mask

    def transform_2d_slice(self, img_slice, seg_slice, seed):
        if self.train:
            img_slice = TF.to_pil_image(img_slice)
            seg_slice = TF.to_pil_image(seg_slice)
            
            random.seed(seed)
            torch.manual_seed(seed)
            img_slice = self.slice_transforms(img_slice)
            
            random.seed(seed)
            torch.manual_seed(seed)
            seg_slice = self.slice_transforms(seg_slice)
            
            img_slice = TF.to_tensor(img_slice)
            seg_slice = TF.to_tensor(seg_slice)
            seg_slice = (seg_slice >= 0.5).float()
            
        return img_slice, seg_slice

    def get_bounding_box(self, seg):
        # Find the indices of non-zero elements
        nonzero = np.nonzero(seg)
        
        # Get the bounding box coordinates
        bbox = np.array([
            [np.min(nonzero[0]), np.max(nonzero[0])],
            [np.min(nonzero[1]), np.max(nonzero[1])],
        ])
        
        return bbox
    
    def crop_center(self, img, bbox, target_shape):
        img = img[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
        img = F.pad(img, (0, target_shape[1] - img.shape[1], 0, target_shape[0] - img.shape[0]))
        return img
    
    def apply_window(self, img):
        if self.train:
            window_center = self.base_window_center + random.uniform(-20, 20)
            window_width = self.base_window_width * random.uniform(0.8, 1.2)
        else:
            window_center = self.base_window_center
            window_width = self.base_window_width

        window_min = window_center - window_width/2
        window_max = window_center + window_width/2
        
        img = (img - window_min) / (window_max - window_min)
        img = torch.clip(img, 0, 1)
        
        return img

    def __getitem__(self, index: int):
        # self.randomize()
        seed = self.rng.randint(2**32)
        self._seed = seed
        meta_data, seg_meta_data, seg, label = None, None, None, None

        # load data and optionally meta
        if self.image_only:
            img = self.loader(self.image_files[index])
            if self.seg_files is not None:
                seg = self.loader(self.seg_files[index])
        else:
            img, meta_data = self.loader(self.image_files[index])
            if self.seg_files is not None:
                seg, seg_meta_data = self.loader(self.seg_files[index])

        # CT liver window
        img = self.apply_window(img)

        seg_mean = seg.mean(axis=0).mean(axis=0)
        pos_idx = torch.where(seg_mean > 0)[0]

        # Dilate the segmentation mask with random kernel size during training
        for p in pos_idx:
            kernel_size = 17
            seg[:, :, p] = torch.nn.functional.max_pool2d(
                seg[:, :, p].unsqueeze(0).unsqueeze(0), 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size//2
            ).squeeze(0).squeeze(0)

        if self.train:
            sample_ratio = random.uniform(0.4, 1.0)
            num_samples = max(2, int(len(pos_idx) * sample_ratio))
            pos_idx = random.sample(pos_idx.tolist(), num_samples)
            pos_idx.sort()
            img = img[:, :, pos_idx]
            seg = seg[:, :, pos_idx]
        else:
            img = img[:, :, pos_idx]
            seg = seg[:, :, pos_idx]

        # apply the transforms
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                img, meta_data = apply_transform(self.transform, (img, meta_data), map_items=False, unpack_items=True)
            else:
                img = apply_transform(self.transform, img, map_items=False)

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                seg, seg_meta_data = apply_transform(
                    self.seg_transform, (seg, seg_meta_data), map_items=False, unpack_items=True
                )
            else:
                seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            label = self.labels[index]
            if self.label_transform is not None:
                label = apply_transform(self.label_transform, label, map_items=False)  # type: ignore

        # Transform 2D slices if in training mode
        if self.train:
            transformed_img_slices = []
            transformed_seg_slices = []
            seed = random.randint(0, 2**32)
            for i in range(img.shape[-1]):
                img_slice = img[..., i]
                seg_slice = seg[..., i]
                img_slice, seg_slice = self.transform_2d_slice(img_slice, seg_slice, seed)
                transformed_img_slices.append(img_slice)
                transformed_seg_slices.append(seg_slice)
            
            img = torch.stack(transformed_img_slices, dim=-1)
            seg = torch.stack(transformed_seg_slices, dim=-1)

        # Pad or crop to max_slices
        img, seg, valid_mask = self.pad_to_max_slices(img, seg)

        img = monai.transforms.Resize((224, 224, img.shape[-1]))(img)
        seg = monai.transforms.Resize((224, 224, seg.shape[-1]))(seg)
        img = img.repeat(3, 1, 1, 1)

        data = [img]
        if seg is not None:
            data.append(seg)
        if label is not None:
            data.append(label)
        if self.rad_feat is not None:
            data.append(self.rad_feat[index])
        # Add valid_mask to the output
        data.append(valid_mask)
        if not self.image_only and meta_data is not None:
            data.append(meta_data)
        if not self.image_only and seg_meta_data is not None:
            data.append(seg_meta_data)

        id = self.image_files[index].split('/')[-2]
        data.append(id)
        if len(data) == 1:
            return data[0]
        return tuple(data) 
