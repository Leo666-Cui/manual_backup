import logging
import os
import sys
import argparse
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import time
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
from pc_grad import PCGrad 


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
    parser.add_argument('--learning_rate', type=float, default=1e-3) # default=1e-3
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
    pfs_df = pd.read_csv('/data1/yxcui/FM-Bridge/Data/GZ-Liver/internal-pfs.csv')
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
        # print("--------------------------------------------------")
        # print(f"警告：发现 {len(missing_pfs_patients)} 个病人缺少 '2-year-PFS' 数据，将被过滤掉：")
        # print(missing_pfs_patients[['name', 'hospital number', 'pathological number', '2-year-PFS']].head())
        # print("--------------------------------------------------")
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
    clip = load_clip_to_cpu('/data1/yxcui/FM-Bridge/testing_file/fm_bridge_new/weights/open_clip_pytorch_model.bin', args.vision_prompt_dep, args.vision_prompt_len)
    model = Model(clip.to(device), ori_clip.to(device), args.text_prompt_len).to(device)

    # Define loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 2]).float().to(device), label_smoothing=0.4)
    # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = PCGrad(base_optimizer)

    # Training loop
    val_interval = 1
    best_hcc_auc = -1
    best_pfs_auc = -1
    best_hcc_epoch = -1
    best_pfs_epoch = -1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    # Extended history for multi-task learning
    history = {
        'train_loss': [], 'train_hcc_auc': [], 'train_pfs_auc': [], 
        'train_hcc_acc': [], 'train_pfs_acc': [], 'train_hcc_f1': [], 'train_pfs_f1': [],
        'val_loss': [], 'val_hcc_auc': [], 'val_pfs_auc': [], 
        'val_hcc_acc': [], 'val_pfs_acc': [], 'val_hcc_f1': [], 'val_pfs_f1': [],
        'best_hcc_auc': [], 'best_pfs_auc': [], 'best_hcc_epoch': [], 'best_pfs_epoch': []
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

        total_data_loading_time = 0
        total_model_training_time = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")
        # Start the timer for the very first batch's data loading
        data_loading_start_time = time.perf_counter()
        for batch_data in train_pbar:
            # The data is ready. Stop the data timer.
            data_loading_end_time = time.perf_counter()
            data_loading_duration = data_loading_end_time - data_loading_start_time
            total_data_loading_time += data_loading_duration

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
            model_step_start_time = time.perf_counter()
            hcc_outputs, pfs_outputs = model(inputs, rad_feat, valid_mask)
            # print(f"Epoch {epoch + 1}, Step {step}: HCC outputs: {hcc_outputs}")
            # print(f"Epoch {epoch + 1}, Step {step}: PFS outputs: {pfs_outputs}")

            # Debug: Check for NaN/inf in model outputs
            # 测试代码
            if torch.isnan(hcc_outputs).any() or torch.isinf(hcc_outputs).any():
                print(f"Warning: NaN/inf in HCC outputs at epoch {epoch + 1}, step {step}")
                print(f"HCC outputs range: {hcc_outputs.min().item():.4f} to {hcc_outputs.max().item():.4f}")
            if torch.isnan(pfs_outputs).any() or torch.isinf(pfs_outputs).any():
                print(f"Warning: NaN/inf in PFS outputs at epoch {epoch + 1}, step {step}")
                print(f"PFS outputs range: {pfs_outputs.min().item():.4f} to {pfs_outputs.max().item():.4f}")

            # Compute losses for both tasks
            hcc_loss = loss_function(hcc_outputs, labels_hcc)
            pfs_loss = loss_function(pfs_outputs, labels_pfs)
            # cprint(f"Epoch {epoch + 1}, Step {step}: HCC loss: {hcc_loss.item():.4f}, PFS loss: {pfs_loss.item():.4f}" , 'red')
            classification_loss = hcc_loss + pfs_loss

            # Debug: Check for NaN in losses
            # 测试代码
            if torch.isnan(hcc_loss) or torch.isnan(pfs_loss):
                print(f"Warning: NaN in losses at epoch {epoch + 1}, step {step}")
                print(f"HCC loss: {hcc_loss.item() if not torch.isnan(hcc_loss) else 'NaN'}")
                print(f"PFS loss: {pfs_loss.item() if not torch.isnan(pfs_loss) else 'NaN'}")
                print(f"HCC outputs sample: {hcc_outputs[0].detach().cpu().numpy()}")
                print(f"PFS outputs sample: {pfs_outputs[0].detach().cpu().numpy()}")
                print(f"HCC labels sample: {labels_hcc[0].item()}")
                print(f"PFS labels sample: {labels_pfs[0].item()}")
                # Skip this problematic batch
                continue
            
            loss = classification_loss + args.contrastive_loss_weight*model.contrastive_loss + args.orthogonal_loss_weight*model.orthogonal_loss
            
            # loss.backward() 
            optimizer.pc_backward([hcc_loss, pfs_loss])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # default=1.0
            optimizer.step()
            
            # Debug: Check for NaN in model parameters after optimizer step
            nan_params = []
            zero_grad_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params.append(name)
                if param.grad is not None and param.grad.norm().item() == 0:
                    zero_grad_params.append(name)
            
            if nan_params:
                print(f"!!! CRITICAL: NaN/Inf detected in parameters after step {step}: {nan_params[:5]} !!!")
                # Stop training if too many parameters have NaN
                # if len(nan_params) > 10:
                #     print("!!! TOO MANY NaN PARAMETERS - STOPPING TRAINING !!!")
                #     break
            
            # Debug: Print zero gradient parameters for first few steps
            # if step <= 3 and zero_grad_params:
            #     print(f"Step {step}: Parameters with zero gradients: {zero_grad_params[:10]}")
            
            # Stop the model timer
            model_step_end_time = time.perf_counter()
            model_training_duration = model_step_end_time - model_step_start_time
            total_model_training_time += model_training_duration
            
            # --- Update the progress bar with live timing info ---
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'DataTime': f"{data_loading_duration:.3f}s",
                'ModelTime': f"{model_training_duration:.3f}s"
            })

            data_loading_start_time = time.perf_counter()

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

        # --- After the epoch is finished, print the average times ---
        avg_data_time = total_data_loading_time
        avg_model_time = total_model_training_time
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Data Loading Time Batch: {avg_data_time:.4f} seconds")
        print(f"  Average Model Training Time Batch: {avg_model_time:.4f} seconds")

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

                # Save best parameters for HCC
                if val_hcc_auc > best_hcc_auc:
                    best_hcc_auc = val_hcc_auc
                    best_hcc_epoch = epoch + 1
                    history['best_hcc_auc'].append(best_hcc_auc)
                    history['best_hcc_epoch'].append(best_hcc_epoch)
                # Save best parameters for PFS
                if val_pfs_auc > best_pfs_auc:
                    best_pfs_auc = val_pfs_auc
                    best_pfs_epoch = epoch + 1
                    history['best_pfs_auc'].append(best_pfs_auc)
                    history['best_pfs_epoch'].append(best_pfs_epoch)

                print(f"current epoch: {epoch + 1}")
                print(f"  HCC - AUC: {val_hcc_auc:.4f}, ACC: {val_hcc_acc:.4f}, F1: {val_hcc_f1:.4f}")
                print(f"  PFS - AUC: {val_pfs_auc:.4f}, ACC: {val_pfs_acc:.4f}, F1: {val_pfs_f1:.4f}")
                print(f"  Average AUC: {avg_auc:.4f}, Best Average AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
                print(f"  Best HCC AUC: {best_hcc_auc:.4f} at epoch {best_hcc_epoch}, Best PFS AUC: {best_pfs_auc:.4f} at epoch {best_pfs_epoch}")

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
                writer.add_scalar("Metrics/best_val_hcc_auc", best_hcc_auc, epoch)
                writer.add_scalar("Metrics/best_val_pfs_auc", best_pfs_auc, epoch)
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
