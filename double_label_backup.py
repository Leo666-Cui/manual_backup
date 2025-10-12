model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from learnable import TextEncoder, PromptLearner
from utils import orthogonal_loss, Prompts
from safetensors.torch import load_file


class Model(nn.Module):
    def __init__(self, encoder, ori_encoder, text_prompt_len):
        super(Model, self).__init__()
        self.encoder = encoder
        self.ori_encoder = ori_encoder
        # freeze the encoder
        for param in self.encoder.parameters():
            if 'VPT' in str(param):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.ori_encoder.parameters():
            param.requires_grad = False

        # Two separate classifiers for multi-task learning
        self.classifier_hcc = nn.Linear(512*2, 2)  # For proliferative HCC prediction
        self.classifier_pfs = nn.Linear(512*2, 2)  # For 2-year PFS prediction
        self.dropout = nn.Dropout(0.1)  # Add dropout layer
        self.relu = nn.ReLU()
        self.norm_image = nn.BatchNorm1d(512*2)
        self.norm_text = nn.BatchNorm1d(512)

        self.text_encoder = TextEncoder(self.ori_encoder)
        self.prompt_learner = nn.ModuleList([PromptLearner(Prompts[i], self.ori_encoder, text_prompt_len, 0, 0, False) for i in range(len(Prompts))])
        self.tokenized_prompts = [learner.tokenized_prompts for learner in self.prompt_learner]
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)  # temperature parameter

        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(512*2)
        )

        self.conv_1d = nn.Conv1d(512, 512, kernel_size=3, padding=1)

    def forward(self, x, rad_feat=None, valid_mask=None):
        # Generate text prompts based on rad_feat values
        seq_len = x.shape[-1]
        prompts = []
        for i in range(len(self.prompt_learner)):
            prompts.append(self.prompt_learner[i]())
        tokenized_prompts = self.tokenized_prompts
        self.text_feats = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(prompts, tokenized_prompts)]

        # Vectorized operation to get feature indices for all samples at once
        feat_indices = rad_feat.long()  # Convert to integer indices
        # Get text features for each radiological feature using 0/1 indices
        text_latents = torch.stack([
            torch.stack([self.text_feats[j][feat_indices[i,j]] for j in range(rad_feat.shape[1])])
            for i in range(rad_feat.shape[0])
        ])
        text_latents = text_latents / text_latents.norm(dim=-1, keepdim=True)
        neg_text_latents = torch.stack([
            torch.stack([self.text_feats[j][0 if feat_indices[i,j] == 1 else 1] for j in range(rad_feat.shape[1])])
            for i in range(rad_feat.shape[0])
        ])
        neg_text_latents = neg_text_latents / neg_text_latents.norm(dim=-1, keepdim=True)

        # Process valid slices only
        valid_idx = torch.where(valid_mask == 1)
        x = x[valid_idx[0], :, :, :, valid_idx[1]]
        image_latents = self.encoder.encode_image(x).float()
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
        image_latents_mean = torch.stack([image_latents[valid_idx[0] == i].mean(dim=0) for i in range(valid_mask.shape[0])])

        # Pad to max length
        max_len = 50
        padded_latents = []
        for i in range(valid_mask.shape[0]):
            curr_latents = image_latents[valid_idx[0] == i]
            pad_len = max_len - len(curr_latents)
            if pad_len > 0:
                padding = torch.zeros(pad_len, curr_latents.shape[-1], device=curr_latents.device)
                curr_latents = torch.cat([curr_latents, padding], dim=0)
            padded_latents.append(curr_latents)
        image_latents = torch.stack(padded_latents)
        
        # Calculate contrastive loss
        pos_sim = torch.einsum('bd,bmd->bm', image_latents_mean.detach(), text_latents)  # shape: (batch_size, num_features)
        neg_sim = torch.einsum('bd,bmd->bm', image_latents_mean.detach(), neg_text_latents)  # shape: (batch_size, num_features)
        logits = torch.stack([pos_sim, neg_sim], dim=2) / self.temperature  # shape: (batch_size, num_features, 2)
        labels = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=logits.device)  # shape: (batch_size, num_features)
        self.contrastive_loss = nn.CrossEntropyLoss()(
            logits.view(-1, 2),  # reshape to (batch_size * num_features, 2)
            labels.view(-1)      # reshape to (batch_size * num_features)
        )

        # Calculate orthogonal loss
        self.orthogonal_loss = orthogonal_loss(torch.cat([text_latents, neg_text_latents], dim=1).transpose(1, 2))

        # Feature pooling with attention
        pooled_image_latents = []
        pooled_text_latents = []
        for i in range(valid_mask.shape[0]):
            valid_features = image_latents[i, valid_mask[i] == 1]
            attention_weights = torch.einsum('md,ld->ml', text_latents[i], valid_features)
            attention_image = attention_weights.mean(dim=0)
            attention_text = attention_weights.mean(dim=-1)
            mean_image_features = torch.einsum('l,ld->d', attention_image, valid_features)
            mean_image_features = mean_image_features / mean_image_features.norm(dim=-1, keepdim=True)
            mean_text_features = torch.einsum('m,md->d', attention_text, text_latents[i])
            mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
            pooled_image_latents.append(mean_image_features)
            pooled_text_latents.append(mean_text_features)
        
        image_latents = torch.stack(pooled_image_latents)  # Shape: (batch, channels)
        text_latents = torch.stack(pooled_text_latents)  # Shape: (batch, channels)
        
        # Combine features and classify
        combined_features = torch.cat([image_latents, text_latents], dim=1)
        latents = self.fusion(combined_features)
        
        # Two separate outputs for multi-task learning
        hcc_output = self.classifier_hcc(latents)
        pfs_output = self.classifier_pfs(latents)
        
        return hcc_output, pfs_output

def load_clip_to_cpu(model_path, vision_prompt_dep, vision_prompt_len):
    """Load CLIP model from path with optional prompt parameters."""
    from clip_custom import clip
    
    # state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = load_file(model_path, device="cpu")
    design_details = {
        "trainer": 'IVLP',
        "vision_depth": vision_prompt_dep,
        "language_depth": 0,
        "vision_ctx": vision_prompt_len,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict, design_details)
    return model 




dataset.py
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
    def __init__(self, image_files, seg_files, labels_hcc, labels_pfs, rad_feat, transform=None, seg_transform=None, train=None):
        # Pass labels_hcc as the main labels to parent class for compatibility
        super().__init__(image_files, seg_files, labels_hcc, transform=transform, seg_transform=seg_transform)
        
        # Store both label types
        self.labels_hcc = labels_hcc
        self.labels_pfs = labels_pfs

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
        meta_data, seg_meta_data, seg, label_hcc, label_pfs = None, None, None, None, None

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
            label_hcc = self.labels_hcc[index]
            label_pfs = self.labels_pfs[index]
            if self.label_transform is not None:
                label_hcc = apply_transform(self.label_transform, label_hcc, map_items=False)  # type: ignore
                label_pfs = apply_transform(self.label_transform, label_pfs, map_items=False)  # type: ignore

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
        # Add both labels for multi-task learning
        if label_hcc is not None:
            data.append(label_hcc)
        if label_pfs is not None:
            data.append(label_pfs)
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



train.py
import logging
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.transforms import EnsureChannelFirst, Compose
from monai.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import open_clip
from datetime import datetime
import wandb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from termcolor import cprint

from dataset import Dataset
from model import Model, load_clip_to_cpu
from utils import set_seed, worker_init_fn, drop_id, session, rad_feat_ind

def plot_and_save_metrics(history, epochs, run_name):
    """根据历史数据绘图并保存为图片 - Multi-task version"""
    # 确保保存结果的文件夹存在
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    epochs_range = range(1, epochs + 1)

    # 1. 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_loss.png')
    plt.close()

    # 2. 绘制 HCC AUC 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_hcc_auc'], label='Training HCC AUC')
    plt.plot(epochs_range, history['val_hcc_auc'], label='Validation HCC AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('HCC Task: Training and Validation AUC')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_hcc_auc.png')
    plt.close()

    # 3. 绘制 PFS AUC 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_pfs_auc'], label='Training PFS AUC')
    plt.plot(epochs_range, history['val_pfs_auc'], label='Validation PFS AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('PFS Task: Training and Validation AUC')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_pfs_auc.png')
    plt.close()

    # 4. 绘制 HCC Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_hcc_acc'], label='Training HCC Accuracy')
    plt.plot(epochs_range, history['val_hcc_acc'], label='Validation HCC Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('HCC Task: Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_hcc_accuracy.png')
    plt.close()

    # 5. 绘制 PFS Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_pfs_acc'], label='Training PFS Accuracy')
    plt.plot(epochs_range, history['val_pfs_acc'], label='Validation PFS Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('PFS Task: Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_pfs_accuracy.png')
    plt.close()

    # 6. 绘制 HCC F1 Score 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_hcc_f1'], label='Training HCC F1 Score')
    plt.plot(epochs_range, history['val_hcc_f1'], label='Validation HCC F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('HCC Task: Training and Validation F1 Score')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_hcc_f1_score.png')
    plt.close()

    # 7. 绘制 PFS F1 Score 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_pfs_f1'], label='Training PFS F1 Score')
    plt.plot(epochs_range, history['val_pfs_f1'], label='Validation PFS F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('PFS Task: Training and Validation F1 Score')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_pfs_f1_score.png')
    plt.close()
    
    print(f"Metrics plots saved to {output_dir}/")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_prompt_len', type=int, default=6)
    parser.add_argument('--vision_prompt_dep', type=int, default=10)
    parser.add_argument('--vision_prompt_len', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--contrastive_loss_weight', type=float, default=0)
    parser.add_argument('--orthogonal_loss_weight', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300) # default: 300
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set random seed for reproducibility
    SEED = args.seed
    set_seed(SEED)

    # Create run name based on parameters
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f'conattn_sepnorm_vdep{args.vision_prompt_dep}_vlen{args.vision_prompt_len}_tlen{args.text_prompt_len}_coloss{args.contrastive_loss_weight}_orloss{args.orthogonal_loss_weight}_dualattn_labelsmooth04_lr{args.learning_rate}_weight12_{timestamp}'

    # Set up tensorboard writer
    writer = SummaryWriter(f'runs_3d/{run_name}')

    # Initialize wandb if specified
    if args.use_wandb:
        wandb.login(key='945c7addff384a579a9bafb12828e3bf1b040d64')
        wandb.init(
            project='GZ-Liver',
            name=run_name,
            config={
                "seed": SEED,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "contrastive_loss_weight": args.contrastive_loss_weight,
                "orthogonal_loss_weight": args.orthogonal_loss_weight,
                "text_prompt_len": args.text_prompt_len,
                "vision_prompt_dep": args.vision_prompt_dep,
                "vision_prompt_len": args.vision_prompt_len,
            }
        )
        wandb.save('train.py')
        wandb.save('model.py')
        wandb.save('dataset.py')
        wandb.save('learnable.py')
        wandb.save('utils.py')

    # Configure logging
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Define data paths
    nii_path = './Data/GZ-Liver/Internal_nii2'
    roi_path = './Data/GZ-Liver/Internal_roi'
    label_df = pd.read_csv('./Data/GZ-Liver/label.csv')
    rad_feat_df = pd.read_csv('./Data/GZ-Liver/rad_cli_feat.csv')
    pfs_df = pd.read_csv('/home/yxcui/FM-Bridge/Data/GZ-Liver/internal-pfs.csv')
    label_df = label_df.merge(rad_feat_df, on=['pathological number', 'label'], how='left')
    
    # 去除'name'列字符串前后可能存在的空格 + 转换为整数
    label_df['name'] = label_df['name'].str.strip()
    label_df['label'] = label_df['label'].astype(int) 
    pfs_df['name'] = pfs_df['name'].str.strip()
    pfs_df['label'] = pfs_df['label'].astype(int)

    final_df = pd.merge(
        label_df, 
        pfs_df[['name', 'label', '2-year-PFS']], 
        on=['name', 'label'], 
        how='left'
    )
    
    # Check and filter out missing PFS data
    missing_pfs_patients = final_df[final_df['2-year-PFS'].isnull()]
    if not missing_pfs_patients.empty:
        print("--------------------------------------------------")
        print(f"警告：发现 {len(missing_pfs_patients)} 个病人缺少 '2-year-PFS' 数据，将被过滤掉：")
        print(missing_pfs_patients[['name', 'hospital number', 'pathological number', '2-year-PFS']].head())
        print("--------------------------------------------------")
        # Filter out rows with missing PFS data
        final_df = final_df[final_df['2-year-PFS'].notna()]
        print(f"过滤后剩余 {len(final_df)} 个样本")
    else:
        print("恭喜！在 '2-year-PFS' 列中没有发现任何缺失值。")
    
    # Ensure PFS labels are integers
    final_df['2-year-PFS'] = final_df['2-year-PFS'].astype(int)

    # Prepare training data
    train_images_id = final_df[final_df['train'] == 1]['hospital number'].values
    train_images_id = [f for f in train_images_id if f not in drop_id]
    train_images = [os.path.join(nii_path, str(f), f'{session}.nii.gz') for f in train_images_id]
    train_segs = [os.path.join(roi_path, str(f), f'{session}.nrrd') for f in train_images_id]

    # Load both HCC labels and PFS labels for multi-task learning
    train_labels_hcc = []
    train_labels_pfs = []
    train_rad_feat = []
    for f in train_images_id:
        train_labels_hcc.append(final_df[final_df['hospital number'] == int(f)]['label'].values[0])
        train_labels_pfs.append(final_df[final_df['hospital number'] == int(f)]['2-year-PFS'].values[0])
        train_rad_feat.append(final_df[final_df['hospital number'] == int(f)][rad_feat_ind].values[0])
    train_labels_hcc = np.array(train_labels_hcc, dtype=np.int64)
    train_labels_pfs = np.array(train_labels_pfs, dtype=np.int64)
    train_rad_feat = np.array(train_rad_feat, dtype=np.float32)

    # Prepare validation data
    valid_images_id = final_df[final_df['train'] == 0]['hospital number'].values
    valid_images_id = [f for f in valid_images_id if f not in drop_id]
    valid_images = [os.path.join(nii_path, str(f), f'{session}.nii.gz') for f in valid_images_id]
    valid_segs = [os.path.join(roi_path, str(f), f'{session}.nrrd') for f in valid_images_id]

    # Load both HCC labels and PFS labels for multi-task learning
    valid_labels_hcc = []
    valid_labels_pfs = []
    valid_rad_feat = []
    for f in valid_images_id:
        valid_labels_hcc.append(final_df[final_df['hospital number'] == int(f)]['label'].values[0])
        valid_labels_pfs.append(final_df[final_df['hospital number'] == int(f)]['2-year-PFS'].values[0])
        valid_rad_feat.append(final_df[final_df['hospital number'] == int(f)][rad_feat_ind].values[0])
    valid_labels_hcc = np.array(valid_labels_hcc, dtype=np.int64)
    valid_labels_pfs = np.array(valid_labels_pfs, dtype=np.int64)
    valid_rad_feat = np.array(valid_rad_feat, dtype=np.float32)

    # Define transforms
    train_transforms = Compose([EnsureChannelFirst()])
    val_transforms = Compose([EnsureChannelFirst()])

    # Create datasets with both labels for multi-task learning
    train_ds = Dataset(train_images, train_segs, train_labels_hcc, train_labels_pfs, train_rad_feat, transform=train_transforms, seg_transform=train_transforms, train=True)
    val_ds = Dataset(valid_images, valid_segs, valid_labels_hcc, valid_labels_pfs, valid_rad_feat, transform=val_transforms, seg_transform=val_transforms, train=False)

    # Create data loaders, num_workers=4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=8, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, drop_last=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    ori_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:wisdomik/GenMedClip')
    clip = load_clip_to_cpu('/home/yxcui/FM-Bridge/testing_file/fm_bridge_new/weights/open_clip_pytorch_model.bin', args.vision_prompt_dep, args.vision_prompt_len)
    model = Model(clip.to(device), ori_clip.to(device), args.text_prompt_len).to(device)

    # Define loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 2]).float().to(device), label_smoothing=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)

    # Training loop
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    # Extended history for multi-task learning
    history = {
        'train_loss': [], 'train_hcc_auc': [], 'train_pfs_auc': [], 
        'train_hcc_acc': [], 'train_pfs_acc': [], 'train_hcc_f1': [], 'train_pfs_f1': [],
        'val_loss': [], 'val_hcc_auc': [], 'val_pfs_auc': [], 
        'val_hcc_acc': [], 'val_pfs_acc': [], 'val_hcc_f1': [], 'val_pfs_f1': []
    }
    
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        epoch_contrastive_loss = 0
        epoch_orthogonal_loss = 0
        step = 0
        # Separate tracking for HCC and PFS tasks
        train_prob_hcc_all, train_label_hcc_all = [], []
        train_prob_pfs_all, train_label_pfs_all = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")
        for batch_data in train_pbar:
            step += 1
            # Unpack both labels
            inputs, segs, labels_hcc, labels_pfs, rad_feat, valid_mask = (
                batch_data[0].to(device), batch_data[1].to(device), 
                batch_data[2].to(device), batch_data[3].to(device),
                batch_data[4].to(device), batch_data[5].to(device)
            )
            
            inputs = inputs * segs
            inputs = inputs.as_tensor()
            optimizer.zero_grad()
            
            # Get both outputs from model
            hcc_outputs, pfs_outputs = model(inputs, rad_feat, valid_mask)
            
            # Compute losses for both tasks
            hcc_loss = loss_function(hcc_outputs, labels_hcc)
            pfs_loss = loss_function(pfs_outputs, labels_pfs)
            classification_loss = hcc_loss + pfs_loss
            
            loss = classification_loss + args.contrastive_loss_weight*model.contrastive_loss + args.orthogonal_loss_weight*model.orthogonal_loss
            
            loss.backward()
            optimizer.step()
            
            # Track predictions for both tasks
            train_prob_hcc = torch.nn.functional.softmax(hcc_outputs, dim=1)
            train_prob_pfs = torch.nn.functional.softmax(pfs_outputs, dim=1)
            train_prob_hcc_all.append(train_prob_hcc.detach().to("cpu").numpy())
            train_prob_pfs_all.append(train_prob_pfs.detach().to("cpu").numpy())
            train_label_hcc_all.append(labels_hcc.to("cpu").numpy())
            train_label_pfs_all.append(labels_pfs.to("cpu").numpy())
            
            epoch_loss += classification_loss.item()
            epoch_contrastive_loss += model.contrastive_loss.item()
            epoch_orthogonal_loss += model.orthogonal_loss.item()
            train_pbar.set_postfix(loss=classification_loss.item())

        # Calculate training metrics for both tasks
        epoch_loss /= step
        history['train_loss'].append(epoch_loss)
        epoch_contrastive_loss /= step
        epoch_orthogonal_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        # HCC task metrics
        train_prob_hcc_all = np.concatenate(train_prob_hcc_all)
        train_label_hcc_all = np.concatenate(train_label_hcc_all)
        train_hcc_auc = roc_auc_score(train_label_hcc_all, train_prob_hcc_all[:, 1])
        train_hcc_acc = accuracy_score(train_label_hcc_all, train_prob_hcc_all[:, 1].round())
        train_hcc_f1 = f1_score(train_label_hcc_all, train_prob_hcc_all[:, 1].round())
        history['train_hcc_auc'].append(train_hcc_auc)
        history['train_hcc_acc'].append(train_hcc_acc)
        history['train_hcc_f1'].append(train_hcc_f1)
        
        # PFS task metrics
        train_prob_pfs_all = np.concatenate(train_prob_pfs_all)
        train_label_pfs_all = np.concatenate(train_label_pfs_all)
        train_pfs_auc = roc_auc_score(train_label_pfs_all, train_prob_pfs_all[:, 1])
        train_pfs_acc = accuracy_score(train_label_pfs_all, train_prob_pfs_all[:, 1].round())
        train_pfs_f1 = f1_score(train_label_pfs_all, train_prob_pfs_all[:, 1].round())
        history['train_pfs_auc'].append(train_pfs_auc)
        history['train_pfs_acc'].append(train_pfs_acc)
        history['train_pfs_f1'].append(train_pfs_f1)
        
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, contrastive loss: {epoch_contrastive_loss:.4f}, orthogonal loss: {epoch_orthogonal_loss:.4f}")
        print(f"  HCC - AUC: {train_hcc_auc:.4f}, ACC: {train_hcc_acc:.4f}, F1: {train_hcc_f1:.4f}")
        print(f"  PFS - AUC: {train_pfs_auc:.4f}, ACC: {train_pfs_acc:.4f}, F1: {train_pfs_f1:.4f}")

        # Log training metrics
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/train_contrastive", epoch_contrastive_loss, epoch)
        writer.add_scalar("Metrics/train_hcc_auc", train_hcc_auc, epoch)
        writer.add_scalar("Metrics/train_hcc_acc", train_hcc_acc, epoch)
        writer.add_scalar("Metrics/train_hcc_f1", train_hcc_f1, epoch)
        writer.add_scalar("Metrics/train_pfs_auc", train_pfs_auc, epoch)
        writer.add_scalar("Metrics/train_pfs_acc", train_pfs_acc, epoch)
        writer.add_scalar("Metrics/train_pfs_f1", train_pfs_f1, epoch)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            # Separate tracking for HCC and PFS validation
            val_prob_hcc_list, val_label_hcc_list = [], []
            val_prob_pfs_list, val_label_pfs_list = [], []
            
            with torch.no_grad():
                val_epoch_loss = 0
                step = 0
                image_id_list = []
                
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]")
                for val_data in val_pbar:
                    step += 1
                    # Unpack both labels
                    val_images, segs, val_labels_hcc, val_labels_pfs, rad_feat, valid_mask, image_id = (
                        val_data[0].to(device), val_data[1].to(device), 
                        val_data[2].to(device), val_data[3].to(device),
                        val_data[4].to(device), val_data[5].to(device), val_data[6]
                    )
                    image_id_list.extend(image_id)
                    val_images = val_images * segs
                    val_images = val_images.as_tensor()
                    
                    # Get both outputs
                    hcc_outputs, pfs_outputs = model(val_images, rad_feat, valid_mask)
                    
                    # Compute losses for both tasks
                    hcc_val_loss = loss_function(hcc_outputs, val_labels_hcc)
                    pfs_val_loss = loss_function(pfs_outputs, val_labels_pfs)
                    val_loss = hcc_val_loss + pfs_val_loss
                    val_epoch_loss += val_loss.item()
                    
                    # Track probabilities for both tasks
                    val_prob_hcc = torch.nn.functional.softmax(hcc_outputs, dim=1)
                    val_prob_pfs = torch.nn.functional.softmax(pfs_outputs, dim=1)
                    val_prob_hcc_list.append(val_prob_hcc.to("cpu").numpy())
                    val_prob_pfs_list.append(val_prob_pfs.to("cpu").numpy())
                    val_label_hcc_list.append(val_labels_hcc.to("cpu").numpy())
                    val_label_pfs_list.append(val_labels_pfs.to("cpu").numpy())
                    
                    val_pbar.set_postfix(val_loss=val_loss.item())
                
                # Calculate validation metrics for both tasks
                val_epoch_loss /= step
                print(f"epoch {epoch + 1} validation loss: {val_epoch_loss:.4f}")
                history['val_loss'].append(val_epoch_loss)
                
                # HCC task metrics
                val_prob_hcc = np.concatenate(val_prob_hcc_list)
                val_label_hcc = np.concatenate(val_label_hcc_list)
                val_hcc_auc = roc_auc_score(val_label_hcc, val_prob_hcc[:, 1])
                val_hcc_acc = accuracy_score(val_label_hcc, val_prob_hcc[:, 1].round())
                val_hcc_f1 = f1_score(val_label_hcc, val_prob_hcc[:, 1].round())
                history['val_hcc_auc'].append(val_hcc_auc)
                history['val_hcc_acc'].append(val_hcc_acc)
                history['val_hcc_f1'].append(val_hcc_f1)
                
                # PFS task metrics
                val_prob_pfs = np.concatenate(val_prob_pfs_list)
                val_label_pfs = np.concatenate(val_label_pfs_list)
                val_pfs_auc = roc_auc_score(val_label_pfs, val_prob_pfs[:, 1])
                val_pfs_acc = accuracy_score(val_label_pfs, val_prob_pfs[:, 1].round())
                val_pfs_f1 = f1_score(val_label_pfs, val_prob_pfs[:, 1].round())
                history['val_pfs_auc'].append(val_pfs_auc)
                history['val_pfs_acc'].append(val_pfs_acc)
                history['val_pfs_f1'].append(val_pfs_f1)
                
                # Average AUC for model selection
                avg_auc = (val_hcc_auc + val_pfs_auc) / 2
                
                # Save best model based on average AUC
                if avg_auc > best_metric:
                    best_metric = avg_auc
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                    print("saved new best metric model")

                    # Create results directory if it doesn't exist
                    os.makedirs('./results', exist_ok=True)
                    
                    # Create DataFrame with results for both tasks
                    results_df = pd.DataFrame({
                        'image_id': image_id_list,
                        'prob_hcc': val_prob_hcc[:, 1],  # Probability of positive class for HCC
                        'label_hcc': val_label_hcc,
                        'prob_pfs': val_prob_pfs[:, 1],  # Probability of positive class for PFS
                        'label_pfs': val_label_pfs
                    })
                    
                    # Save to CSV using run name
                    results_df.to_csv(f'./results/{run_name}.csv', index=False)

                print(f"current epoch: {epoch + 1}")
                print(f"  HCC - AUC: {val_hcc_auc:.4f}, ACC: {val_hcc_acc:.4f}, F1: {val_hcc_f1:.4f}")
                print(f"  PFS - AUC: {val_pfs_auc:.4f}, ACC: {val_pfs_acc:.4f}, F1: {val_pfs_f1:.4f}")
                print(f"  Average AUC: {avg_auc:.4f}, Best Average AUC: {best_metric:.4f} at epoch {best_metric_epoch}")

                # Log validation metrics
                writer.add_scalar("Loss/val", val_epoch_loss, epoch)
                writer.add_scalar("Metrics/val_hcc_auc", val_hcc_auc, epoch)
                writer.add_scalar("Metrics/val_hcc_acc", val_hcc_acc, epoch)
                writer.add_scalar("Metrics/val_hcc_f1", val_hcc_f1, epoch)
                writer.add_scalar("Metrics/val_pfs_auc", val_pfs_auc, epoch)
                writer.add_scalar("Metrics/val_pfs_acc", val_pfs_acc, epoch)
                writer.add_scalar("Metrics/val_pfs_f1", val_pfs_f1, epoch)
                writer.add_scalar("Metrics/val_avg_auc", avg_auc, epoch)
                writer.add_scalar("Metrics/best_val_avg_auc", best_metric, epoch)

                # Log metrics to wandb (inside validation block)
                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "train_contrastive_loss": epoch_contrastive_loss,
                        "train_orthogonal_loss": epoch_orthogonal_loss,
                        "train_hcc_auc": train_hcc_auc,
                        "train_hcc_acc": train_hcc_acc,
                        "train_hcc_f1": train_hcc_f1,
                        "train_pfs_auc": train_pfs_auc,
                        "train_pfs_acc": train_pfs_acc,
                        "train_pfs_f1": train_pfs_f1,
                        "val_loss": val_epoch_loss,
                        "val_hcc_auc": val_hcc_auc,
                        "val_hcc_acc": val_hcc_acc,
                        "val_hcc_f1": val_hcc_f1,
                        "val_pfs_auc": val_pfs_auc,
                        "val_pfs_acc": val_pfs_acc,
                        "val_pfs_f1": val_pfs_f1,
                        "val_avg_auc": avg_auc,
                        "best_val_avg_auc": best_metric
                    })

                    if avg_auc > best_metric:
                        wandb.run.summary["best_val_avg_auc"] = best_metric
                        wandb.run.summary["best_epoch"] = best_metric_epoch

    # Final summary
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    # Record final results
    writer.add_hparams(
        {"lr": args.learning_rate, "batch_size": args.batch_size},
        {
            "best_val_avg_auc": best_metric,
            "best_epoch": best_metric_epoch,
        }
    )
    plot_and_save_metrics(history, args.epochs, run_name)

    if args.use_wandb:
        wandb.run.summary["final_best_avg_auc"] = best_metric
        wandb.run.summary["final_best_epoch"] = best_metric_epoch
        wandb.finish()

    writer.close()

if __name__ == "__main__":
    main() 
