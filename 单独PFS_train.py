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
    """根据历史数据绘图并保存为图片"""
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

    # 2. 绘制 AUC 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_auc'], label='Training AUC')
    plt.plot(epochs_range, history['val_auc'], label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_auc.png')
    plt.close()

    # 3. 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_accuracy.png')
    plt.close()

    # 4. 绘制 F1 Score 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history['train_f1'], label='Training F1 Score')
    plt.plot(epochs_range, history['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/{run_name}_f1_score.png')
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
    final_df.to_csv('/home/yxcui/FM-Bridge/Data/GZ-Liver/final_df.csv', index=False)


    missing_pfs_patients = final_df[final_df['2-year-PFS'].isnull()]
    # 检查筛选结果
    if not missing_pfs_patients.empty:
        print("--------------------------------------------------")
        print("找到了以下 【缺少 '2-year-PFS' 数据】 的病人：")
        
        # 为了方便查看，我们只打印出关键信息
        print(missing_pfs_patients[['name', 'hospital number', 'pathological number', '2-year-PFS']])
        print("--------------------------------------------------")
    else:
        print("恭喜！在 '2-year-PFS' 列中没有发现任何缺失值。")


    # Prepare training data
    train_images_id = final_df[final_df['train'] == 1]['hospital number'].values
    train_images_id = [f for f in train_images_id if f not in drop_id]
    train_images = [os.path.join(nii_path, str(f), f'{session}.nii.gz') for f in train_images_id]
    train_segs = [os.path.join(roi_path, str(f), f'{session}.nrrd') for f in train_images_id]

    train_labels = []
    train_rad_feat = []
    for f in train_images_id:
        # label = 2-year-PFS
        # train_labels.append(label_df[label_df['hospital number'] == int(f)]['label'].values[0])
        train_labels.append(final_df[final_df['hospital number'] == int(f)]['2-year-PFS'].values[0])
        train_rad_feat.append(final_df[final_df['hospital number'] == int(f)][rad_feat_ind].values[0])
    train_labels = np.array(train_labels, dtype=np.int64)
    train_rad_feat = np.array(train_rad_feat, dtype=np.float32)

    # Prepare validation data
    valid_images_id = final_df[final_df['train'] == 0]['hospital number'].values
    valid_images_id = [f for f in valid_images_id if f not in drop_id]
    valid_images = [os.path.join(nii_path, str(f), f'{session}.nii.gz') for f in valid_images_id]
    valid_segs = [os.path.join(roi_path, str(f), f'{session}.nrrd') for f in valid_images_id]

    valid_labels = []
    valid_rad_feat = []
    for f in valid_images_id:
        # valid_labels.append(label_df[label_df['hospital number'] == int(f)]['label'].values[0])
        valid_labels.append(final_df[final_df['hospital number'] == int(f)]['2-year-PFS'].values[0])
        valid_rad_feat.append(final_df[final_df['hospital number'] == int(f)][rad_feat_ind].values[0])
    valid_labels = np.array(valid_labels, dtype=np.int64)
    valid_rad_feat = np.array(valid_rad_feat, dtype=np.float32)

    # Define transforms
    train_transforms = Compose([EnsureChannelFirst()])
    val_transforms = Compose([EnsureChannelFirst()])

    # Create datasets
    train_ds = Dataset(train_images, train_segs, train_labels, train_rad_feat, transform=train_transforms, seg_transform=train_transforms, train=True)
    val_ds = Dataset(valid_images, valid_segs, valid_labels, valid_rad_feat, transform=val_transforms, seg_transform=val_transforms, train=False)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, drop_last=True)

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
    history = {
        'train_loss': [], 'train_auc': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        epoch_contrastive_loss = 0
        epoch_orthogonal_loss = 0
        step = 0
        train_prob_all, train_label_all = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")
        for batch_data in train_pbar:
            step += 1
            inputs, segs, labels, rad_feat, valid_mask = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device), batch_data[3].to(device), batch_data[4].to(device)
            
            inputs = inputs * segs
            inputs = inputs.as_tensor()
            optimizer.zero_grad()
            outputs = model(inputs, rad_feat, valid_mask)
            
            classification_loss = loss_function(outputs, labels)
            loss = classification_loss + args.contrastive_loss_weight*model.contrastive_loss + args.orthogonal_loss_weight*model.orthogonal_loss
            
            loss.backward()
            optimizer.step()
            
            train_prob = torch.nn.functional.softmax(outputs, dim=1)
            train_prob_all.append(train_prob.detach().to("cpu").numpy())
            train_label_all.append(labels.to("cpu").numpy())
            epoch_loss += classification_loss.item()
            epoch_contrastive_loss += model.contrastive_loss.item()
            epoch_orthogonal_loss += model.orthogonal_loss.item()
            train_pbar.set_postfix(loss=classification_loss.item())

        # Calculate training metrics
        epoch_loss /= step
        history['train_loss'].append(epoch_loss) # 记录训练loss
        epoch_contrastive_loss /= step
        epoch_orthogonal_loss /= step
        epoch_loss_values.append(epoch_loss)
        train_prob_all = np.concatenate(train_prob_all)
        train_label_all = np.concatenate(train_label_all)
        train_auc = roc_auc_score(train_label_all, train_prob_all[:, 1])
        history['train_auc'].append(train_auc) # 记录训练AUC
        train_acc = accuracy_score(train_label_all, train_prob_all[:, 1].round())
        history['train_acc'].append(train_acc) # 记录训练ACC
        train_f1 = f1_score(train_label_all, train_prob_all[:, 1].round())
        history['train_f1'].append(train_f1) # 记录训练F1
        
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, contrastive loss: {epoch_contrastive_loss:.4f}, orthogonal loss: {epoch_orthogonal_loss:.4f}, train_auc: {train_auc:.4f}, train_acc: {train_acc:.4f}, train_f1: {train_f1:.4f}")

        # Log training metrics
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/train_contrastive", epoch_contrastive_loss, epoch)
        writer.add_scalar("Metrics/train_auc", train_auc, epoch)
        writer.add_scalar("Metrics/train_acc", train_acc, epoch)
        writer.add_scalar("Metrics/train_f1", train_f1, epoch)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_prob_all_list, val_label_list = [], []
            
            with torch.no_grad():
                val_epoch_loss = 0
                step = 0
                metric_count_all = 0
                num_correct_all = 0
                image_id_list = []
                
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]")
                for val_data in val_pbar:
                    step += 1
                    val_images, segs, val_labels, rad_feat, valid_mask, image_id = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device), val_data[3].to(device), val_data[4].to(device), val_data[5]
                    image_id_list.extend(image_id)
                    val_images = val_images * segs
                    val_images = val_images.as_tensor()
                    val_outputs = model(val_images, rad_feat, valid_mask)
                    pred_all = val_outputs
                    val_loss = loss_function(pred_all, val_labels)
                    val_epoch_loss += val_loss.item()
                    val_prob_all = torch.nn.functional.softmax(pred_all, dim=1)
                    val_prob_all_list.append(val_prob_all.to("cpu").numpy())
                    val_label_list.append(val_labels.to("cpu").numpy())
                    value_all = torch.eq(pred_all.argmax(dim=1), val_labels)
                    metric_count_all += len(value_all)
                    num_correct_all += value_all.sum().item()
                    val_pbar.set_postfix(val_loss=val_loss.item())
                
                # Calculate validation metrics
                val_epoch_loss /= step
                print(f"epoch {epoch + 1} validation loss: {val_epoch_loss:.4f}")
                history['val_loss'].append(val_epoch_loss) # 记录验证loss
                val_prob_all = np.concatenate(val_prob_all_list)
                val_label = np.concatenate(val_label_list)
                auc_all = roc_auc_score(val_label, val_prob_all[:, 1])
                history['val_auc'].append(auc_all) # 记录验证AUC
                acc_all = num_correct_all / metric_count_all
                history['val_acc'].append(acc_all) # 记录验证ACC
                f1_all = f1_score(val_label, val_prob_all[:, 1].round())
                history['val_f1'].append(f1_all) # 记录验证F1
                
                # Save best model
                if auc_all > best_metric:
                    best_metric = auc_all
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                    print("saved new best metric model")

                    # Create results directory if it doesn't exist
                    os.makedirs('./results', exist_ok=True)
                    
                    # Create DataFrame with results,best result's prob for all val patients
                    results_df = pd.DataFrame({
                        'image_id': image_id_list,
                        'prob': val_prob_all[:, 1],  # Probability of positive class
                        'label': val_label
                    })
                    
                    # Save to CSV using run name
                    results_df.to_csv(f'./results/{run_name}.csv', index=False)

                print(
                    "current epoch: {} current auc: {:.4f} current acc: {:.4f} current f1: {:.4f}. best AUC: {:.4f} at epoch {}".format(
                        epoch + 1, auc_all, acc_all, f1_all, best_metric, best_metric_epoch
                    )
                )

                # Log validation metrics
                writer.add_scalar("Loss/val", val_epoch_loss, epoch)
                writer.add_scalar("Metrics/val_auc", auc_all, epoch)
                writer.add_scalar("Metrics/val_acc", acc_all, epoch)
                writer.add_scalar("Metrics/val_f1", f1_all, epoch)
                writer.add_scalar("Metrics/best_val_auc", best_metric, epoch)

        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_contrastive_loss": epoch_contrastive_loss,
                "train_orthogonal_loss": epoch_orthogonal_loss,
                "train_auc": train_auc,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_epoch_loss,
                "val_auc": auc_all,
                "val_acc": acc_all,
                "val_f1": f1_all,
                "best_val_auc": best_metric
            })

            if auc_all > best_metric:
                wandb.run.summary["best_val_auc_all"] = best_metric
                wandb.run.summary["best_epoch_all"] = best_metric_epoch

    # Final summary
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    # Record final results
    writer.add_hparams(
        {"lr": args.learning_rate, "batch_size": args.batch_size},
        {
            "best_val_auc": best_metric,
            "best_epoch": best_metric_epoch,
        }
    )
    plot_and_save_metrics(history, args.epochs, run_name)

    if args.use_wandb:
        wandb.run.summary["final_best_auc_all"] = best_metric
        wandb.run.summary["final_best_epoch_all"] = best_metric_epoch
        wandb.finish()

    writer.close()

if __name__ == "__main__":
    main() 
