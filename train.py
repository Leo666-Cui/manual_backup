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
from typing import List
import json

from dataset import PatientMultiSliceDataset
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

def build_image_list_from_json(json_file_path, base_image_dir):
    """
    根据JSON指令，构建一个图像路径列表 (例如 train_images)。
    文件名格式: {ID}_{时期}_windowed_{切片}.png
    """
    # print(f"\n--- 开始构建 [图像] 列表 ---")
    # 调用通用的构建函数，传入图像特定的文件名格式
    return _build_list_from_json_base(
        json_file_path,
        base_image_dir,
        file_format_string="windowed"
    )

def build_seg_list_from_json(json_file_path, base_seg_dir):
    """
    根据JSON指令，构建一个分割掩码路径列表 (例如 train_segs)。
    文件名格式: {ID}_{时期}_mask_{切片}.png
    """
    # print(f"\n--- 开始构建 [分割掩码] 列表 ---")
    # 调用通用的构建函数，传入分割掩码特定的文件名格式
    return _build_list_from_json_base(
        json_file_path,
        base_seg_dir,
        file_format_string="mask"
    )

def _build_list_from_json_base(json_file_path, base_dir, file_format_string):
    """
    一个通用的基础函数，用于根据JSON指令构建文件路径列表。
    """
    PHASES = ['ap', 'dp', 'pvp']

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到JSON文件 '{json_file_path}'。")
        return []
    except json.JSONDecodeError:
        print(f"错误: JSON文件 '{json_file_path}' 格式不正确。")
        return []

    final_file_list = []
    sorted_patient_ids = sorted(data.keys())
    sorted_patient_ids = [f for f in sorted_patient_ids if f not in drop_id]

    for patient_id in sorted_patient_ids:
        patient_specific_paths = []
        slice_info = data[patient_id]
        slice_numbers = sorted(slice_info.keys(), key=int)
        
        all_files_found = True
        
        for slice_num in slice_numbers:
            for phase in PHASES:
                # 使用传入的文件名格式字符串来构建文件名
                filename = f"{patient_id}_{phase}_{file_format_string}_{slice_num}.png"
                full_path = os.path.join(base_dir, filename)
                
                if not os.path.exists(full_path):
                    print(f"警告: 找不到文件 '{full_path}'。将跳过病人 {patient_id}。")
                    all_files_found = False
                    break
                
                patient_specific_paths.append(full_path)
            
            if not all_files_found:
                break
        
        if all_files_found:
            final_file_list.append(patient_specific_paths)

    # print(f"成功为 {len(final_file_list)} 个病人构建了列表。")
    return final_file_list


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_prompt_len', type=int, default=6)
    parser.add_argument('--vision_prompt_dep', type=int, default=10)
    parser.add_argument('--vision_prompt_len', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--contrastive_loss_weight', type=float, default=0)
    parser.add_argument('--orthogonal_loss_weight', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2) # default=32
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
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Define data paths
    nii_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/img_encoder_nii'
    roi_path = '/home/yxcui/FM-Bridge/testing_file/test_dataset/img_encoder_roi'
    agent_train_json_path = '/home/yxcui/FM-Bridge/testing_file/fine_grained_train_results.json'
    agent_val_json_path = '/home/yxcui/FM-Bridge/testing_file/fine_grained_val_results.json'
    label_df = pd.read_csv('/home/yxcui/FM-Bridge/testing_file/label_df.csv')
    PHASES = ['ap', 'dp', 'pvp']

    # Hyperparameters
    D_MODEL = 512                  # Feature dimension from your encoder (e.g., CLIP's ViT-B/32)
    NUM_SLICES = 5                 # S: Number of primary slices per patient
    NUM_TIME_POINTS = 3            # T: Number of time points (phases) per slice
    NUM_CO_ATTENTION_LAYERS = 4    # N: Number of co-attention layers to stack
    NUM_HEADS = 8                  # Number of heads in MultiheadAttention
    D_FFN = 2048                   # Dimension of the feed-forward network in the transformer layers
    DROPOUT = 0.1 

    # Prepare training data
    with open(agent_train_json_path, 'r') as f:
        train_data = json.load(f)
    train_images_id = sorted(list(train_data.keys()))
    # train_images_id = label_df[label_df['train'] == 1]['hospital number'].values
    train_images_id = [f for f in train_images_id if f not in drop_id]
    # 1. 生成 train_images和train_segs 列表
    train_images = build_image_list_from_json(json_file_path=agent_train_json_path, base_image_dir=nii_path)
    train_segs = build_seg_list_from_json(json_file_path=agent_train_json_path, base_seg_dir=roi_path)
    print(f"训练集病人数量: {len(train_images)}")
    print(f"train_images: \n {train_images}")
    train_labels = []
    # train_rad_feat = []
    for f in train_images_id:
        train_labels.append(label_df[label_df['hospital number'] == int(f)]['label'].values[0])
        # train_rad_feat.append(label_df[label_df['hospital number'] == int(f)][rad_feat_ind].values[0])
    train_labels = np.array(train_labels, dtype=np.int64)
    print(f"train_labels: {train_labels}")
    # train_rad_feat = np.array(train_rad_feat, dtype=np.float32)

    # 问题array
    # 使用三层嵌套的列表推导式，同时在每一层进行排序
    all_train_answers_list = [
        [ # <-- 病人层 (外层)
            [ # <-- 切片层 (中层)
                pattern_info[pattern_key]['answer']
                # 模式层 (内层): 按 'pattern_X' 中的数字排序
                for pattern_key in sorted(pattern_info.keys(), key=lambda p: int(p.split('_')[1]))
            ]
            # 切片层: 按切片号 (整数) 排序
            for slice_key, pattern_info in sorted(train_data[train_images_id].items(), key=lambda item: int(item[0]))
        ]
        # 病人层: 按病人ID排序
        for train_images_id in train_images_id
    ]
    # --- 3. 转换为NumPy数组并返回 ---
    train_rad_feat = np.array(all_train_answers_list)
    print(f'train_rad_feat.shape: {train_rad_feat.shape}')
    print(f'train_rad_feat: \n{train_rad_feat}')


    # Prepare validation data
    with open(agent_val_json_path, 'r') as f:
        val_data = json.load(f)
    val_images_id = list(val_data.keys())
    val_images_id = [f for f in val_images_id if f not in drop_id]
    # valid_images_id = label_df[label_df['train'] == 0]['hospital number'].values
    valid_images = build_image_list_from_json(json_file_path=agent_val_json_path, base_image_dir=nii_path)
    valid_segs = build_seg_list_from_json(json_file_path=agent_val_json_path, base_seg_dir=roi_path)

    valid_labels = []
    # valid_rad_feat = []
    for f in val_images_id:
        valid_labels.append(label_df[label_df['hospital number'] == int(f)]['label'].values[0])
        # valid_rad_feat.append(label_df[label_df['hospital number'] == int(f)][rad_feat_ind].values[0])
    valid_labels = np.array(valid_labels, dtype=np.int64)
    # valid_rad_feat = np.array(valid_rad_feat, dtype=np.float32)

    all_val_answers_list = [
        [ 
            [ 
                pattern_info[pattern_key]['answer']
                for pattern_key in sorted(pattern_info.keys(), key=lambda p: int(p.split('_')[1]))
            ]
            for slice_key, pattern_info in sorted(val_data[val_images_id].items(), key=lambda item: int(item[0]))
        ]
        for val_images_id in val_images_id
    ]
    valid_rad_feat = np.array(all_val_answers_list)
    print(f'valid_rad_feat.shape: {valid_rad_feat.shape}')
    print(f'valid_rad_feat: \n{valid_rad_feat}')

    # Define transforms
    train_transforms = Compose([EnsureChannelFirst()])
    val_transforms = Compose([EnsureChannelFirst()])

    # Create datasets
    # train_images: list of list, 里面的list是病人切片的15张照片路径
    train_ds = PatientMultiSliceDataset(train_images, train_segs, train_labels, train_rad_feat, transform=train_transforms, seg_transform=train_transforms, train=True)
    val_ds = PatientMultiSliceDataset(valid_images, valid_segs, valid_labels, valid_rad_feat, transform=val_transforms, seg_transform=val_transforms, train=False)

    # Create data loaders: 
    # 图像批次维度: torch.Size([1, 1, 224, 224, 15])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, drop_last=True)

    # --- 4. 测试DataLoader ---
    print("从DataLoader中取出一批数据进行测试...")
    batch_data = next(iter(train_loader))
    batch_img, batch_seg, batch_label, batch_rad, batch_id = batch_data

    print(f"图像批次维度: {batch_img.shape}")      # 应该为 torch.Size([8, 1, 224, 224])
    print(f"掩码批次维度: {batch_seg.shape}")      # 应该为 torch.Size([8, 1, 224, 224])
    print(f"标签批次维度: {batch_label.shape}")    # 应该为 torch.Size([8, 9])
    print(f"特征批次维度: {batch_rad.shape}")    # 应该为 torch.Size([8, 9])
    print(f"批次中的ID示例: {batch_id}")      # 打印第一个样本的ID

    # ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    ori_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:wisdomik/GenMedClip')
    clip = load_clip_to_cpu('./weights/open_clip_pytorch_model.bin', args.vision_prompt_dep, args.vision_prompt_len)
    # model = Model(clip.to(device), ori_clip.to(device), args.text_prompt_len).to(device)
    model = Model(
        encoder=clip.to(device),
        ori_encoder=ori_clip.to(device),
        text_prompt_len=args.text_prompt_len,
        num_slices=NUM_SLICES,
        num_time_points=NUM_TIME_POINTS,
        num_co_attention_layers=NUM_CO_ATTENTION_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ffn=D_FFN,
        dropout=DROPOUT
    ).to(device)

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
            inputs, segs, labels, rad_feat = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device), batch_data[3].to(device)
            inputs = inputs * segs
            inputs = inputs.as_tensor()
            optimizer.zero_grad()
            outputs = model(inputs, rad_feat)
            
    #         classification_loss = loss_function(outputs, labels)
    #         loss = classification_loss + args.contrastive_loss_weight*model.contrastive_loss + args.orthogonal_loss_weight*model.orthogonal_loss
            
    #         loss.backward()
    #         optimizer.step()
            
    #         train_prob = torch.nn.functional.softmax(outputs, dim=1)
    #         train_prob_all.append(train_prob.detach().to("cpu").numpy())
    #         train_label_all.append(labels.to("cpu").numpy())
    #         epoch_loss += classification_loss.item()
    #         epoch_contrastive_loss += model.contrastive_loss.item()
    #         epoch_orthogonal_loss += model.orthogonal_loss.item()
    #         train_pbar.set_postfix(loss=classification_loss.item())

    #     # Calculate training metrics
    #     epoch_loss /= step
    #     history['train_loss'].append(epoch_loss) # 记录训练loss
    #     epoch_contrastive_loss /= step
    #     epoch_orthogonal_loss /= step
    #     epoch_loss_values.append(epoch_loss)
    #     train_prob_all = np.concatenate(train_prob_all)
    #     train_label_all = np.concatenate(train_label_all)
    #     train_auc = roc_auc_score(train_label_all, train_prob_all[:, 1])
    #     history['train_auc'].append(train_auc) # 记录训练AUC
    #     train_acc = accuracy_score(train_label_all, train_prob_all[:, 1].round())
    #     history['train_acc'].append(train_acc) # 记录训练ACC
    #     train_f1 = f1_score(train_label_all, train_prob_all[:, 1].round())
    #     history['train_f1'].append(train_f1) # 记录训练F1
        
    #     print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, contrastive loss: {epoch_contrastive_loss:.4f}, orthogonal loss: {epoch_orthogonal_loss:.4f}, train_auc: {train_auc:.4f}, train_acc: {train_acc:.4f}, train_f1: {train_f1:.4f}")

    #     # Log training metrics
    #     writer.add_scalar("Loss/train", epoch_loss, epoch)
    #     writer.add_scalar("Loss/train_contrastive", epoch_contrastive_loss, epoch)
    #     writer.add_scalar("Metrics/train_auc", train_auc, epoch)
    #     writer.add_scalar("Metrics/train_acc", train_acc, epoch)
    #     writer.add_scalar("Metrics/train_f1", train_f1, epoch)

    #     # Validation
    #     if (epoch + 1) % val_interval == 0:
    #         model.eval()
    #         val_prob_all_list, val_label_list = [], []
            
    #         with torch.no_grad():
    #             val_epoch_loss = 0
    #             step = 0
    #             metric_count_all = 0
    #             num_correct_all = 0
    #             image_id_list = []
                
    #             val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]")
    #             for val_data in val_pbar:
    #                 step += 1
    #                 val_images, segs, val_labels, rad_feat, valid_mask, image_id = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device), val_data[3].to(device), val_data[4].to(device), val_data[5]
    #                 image_id_list.extend(image_id)
    #                 val_images = val_images * segs
    #                 val_images = val_images.as_tensor()
    #                 val_outputs = model(val_images, rad_feat, valid_mask)
    #                 pred_all = val_outputs
    #                 val_loss = loss_function(pred_all, val_labels)
    #                 val_epoch_loss += val_loss.item()
    #                 val_prob_all = torch.nn.functional.softmax(pred_all, dim=1)
    #                 val_prob_all_list.append(val_prob_all.to("cpu").numpy())
    #                 val_label_list.append(val_labels.to("cpu").numpy())
    #                 value_all = torch.eq(pred_all.argmax(dim=1), val_labels)
    #                 metric_count_all += len(value_all)
    #                 num_correct_all += value_all.sum().item()
    #                 val_pbar.set_postfix(val_loss=val_loss.item())
                
    #             # Calculate validation metrics
    #             val_epoch_loss /= step
    #             print(f"epoch {epoch + 1} validation loss: {val_epoch_loss:.4f}")
    #             history['val_loss'].append(val_epoch_loss) # 记录验证loss
    #             val_prob_all = np.concatenate(val_prob_all_list)
    #             val_label = np.concatenate(val_label_list)
    #             auc_all = roc_auc_score(val_label, val_prob_all[:, 1])
    #             history['val_auc'].append(auc_all) # 记录验证AUC
    #             acc_all = num_correct_all / metric_count_all
    #             history['val_acc'].append(acc_all) # 记录验证ACC
    #             f1_all = f1_score(val_label, val_prob_all[:, 1].round())
    #             history['val_f1'].append(f1_all) # 记录验证F1
                
    #             # Save best model
    #             if auc_all > best_metric:
    #                 best_metric = auc_all
    #                 best_metric_epoch = epoch + 1
    #                 torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
    #                 print("saved new best metric model")

    #                 # Create results directory if it doesn't exist
    #                 os.makedirs('./results', exist_ok=True)
                    
    #                 # Create DataFrame with results,best result's prob for all val patients
    #                 results_df = pd.DataFrame({
    #                     'image_id': image_id_list,
    #                     'prob': val_prob_all[:, 1],  # Probability of positive class
    #                     'label': val_label
    #                 })
                    
    #                 # Save to CSV using run name
    #                 results_df.to_csv(f'./results/{run_name}.csv', index=False)

    #             print(
    #                 "current epoch: {} current auc: {:.4f} current acc: {:.4f} current f1: {:.4f}. best AUC: {:.4f} at epoch {}".format(
    #                     epoch + 1, auc_all, acc_all, f1_all, best_metric, best_metric_epoch
    #                 )
    #             )

    #             # Log validation metrics
    #             writer.add_scalar("Loss/val", val_epoch_loss, epoch)
    #             writer.add_scalar("Metrics/val_auc", auc_all, epoch)
    #             writer.add_scalar("Metrics/val_acc", acc_all, epoch)
    #             writer.add_scalar("Metrics/val_f1", f1_all, epoch)
    #             writer.add_scalar("Metrics/best_val_auc", best_metric, epoch)

    #     # Log metrics to wandb
    #     if args.use_wandb:
    #         wandb.log({
    #             "epoch": epoch + 1,
    #             "train_loss": epoch_loss,
    #             "train_contrastive_loss": epoch_contrastive_loss,
    #             "train_orthogonal_loss": epoch_orthogonal_loss,
    #             "train_auc": train_auc,
    #             "train_acc": train_acc,
    #             "train_f1": train_f1,
    #             "val_loss": val_epoch_loss,
    #             "val_auc": auc_all,
    #             "val_acc": acc_all,
    #             "val_f1": f1_all,
    #             "best_val_auc": best_metric
    #         })

    #         if auc_all > best_metric:
    #             wandb.run.summary["best_val_auc_all"] = best_metric
    #             wandb.run.summary["best_epoch_all"] = best_metric_epoch

    # # Final summary
    # print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    # # Record final results
    # writer.add_hparams(
    #     {"lr": args.learning_rate, "batch_size": args.batch_size},
    #     {
    #         "best_val_auc": best_metric,
    #         "best_epoch": best_metric_epoch,
    #     }
    # )
    # plot_and_save_metrics(history, args.epochs, run_name)

    # if args.use_wandb:
    #     wandb.run.summary["final_best_auc_all"] = best_metric
    #     wandb.run.summary["final_best_epoch_all"] = best_metric_epoch
    #     wandb.finish()

    # writer.close()

if __name__ == "__main__":
    main() 
