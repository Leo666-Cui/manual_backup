import torch
import torch.nn as nn
import torch.nn.functional as F
from learnable import TextEncoder, PromptLearner
from utils import orthogonal_loss, Prompts
from safetensors.torch import load_file

# class Model(nn.Module):
#     def __init__(self, encoder, ori_encoder, text_prompt_len):
#         super(Model, self).__init__()
#         self.encoder = encoder
#         self.ori_encoder = ori_encoder
#         # freeze the encoder
#         for param in self.encoder.parameters():
#             if 'VPT' in str(param):
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#         for param in self.ori_encoder.parameters():
#             param.requires_grad = False

#         self.classifier = nn.Linear(512*2, 2)
#         self.dropout = nn.Dropout(0.1)  # Add dropout layer
#         self.relu = nn.ReLU()
#         self.norm_image = nn.BatchNorm1d(512*2)
#         self.norm_text = nn.BatchNorm1d(512)

#         self.text_encoder = TextEncoder(self.ori_encoder)
#         self.prompt_learner = nn.ModuleList([PromptLearner(Prompts[i], self.ori_encoder, text_prompt_len, 0, 0, False) for i in range(len(Prompts))])
#         self.tokenized_prompts = [learner.tokenized_prompts for learner in self.prompt_learner]
#         for param in self.text_encoder.parameters():
#             param.requires_grad = False

#         self.temperature = nn.Parameter(torch.ones([]) * 0.07)  # temperature parameter

#         self.fusion = nn.Sequential(
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.BatchNorm1d(512*2)
#         )

#         self.conv_1d = nn.Conv1d(512, 512, kernel_size=3, padding=1)

#     def forward(self, x, rad_feat=None, valid_mask=None):
#         # Generate text prompts based on rad_feat values
#         seq_len = x.shape[-1]
#         prompts = []
#         for i in range(len(self.prompt_learner)):
#             prompts.append(self.prompt_learner[i]())
#         tokenized_prompts = self.tokenized_prompts
#         self.text_feats = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(prompts, tokenized_prompts)]

#         # Vectorized operation to get feature indices for all samples at once
#         feat_indices = rad_feat.long()  # Convert to integer indices
#         # Get text features for each radiological feature using 0/1 indices
#         text_latents = torch.stack([
#             torch.stack([self.text_feats[j][feat_indices[i,j]] for j in range(rad_feat.shape[1])])
#             for i in range(rad_feat.shape[0])
#         ])
#         text_latents = text_latents / text_latents.norm(dim=-1, keepdim=True)
#         neg_text_latents = torch.stack([
#             torch.stack([self.text_feats[j][0 if feat_indices[i,j] == 1 else 1] for j in range(rad_feat.shape[1])])
#             for i in range(rad_feat.shape[0])
#         ])
#         neg_text_latents = neg_text_latents / neg_text_latents.norm(dim=-1, keepdim=True)

#         # Process valid slices only
#         valid_idx = torch.where(valid_mask == 1)
#         x = x[valid_idx[0], :, :, :, valid_idx[1]]
#         image_latents = self.encoder.encode_image(x).float()
#         image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
#         image_latents_mean = torch.stack([image_latents[valid_idx[0] == i].mean(dim=0) for i in range(valid_mask.shape[0])])

#         # Pad to max length
#         max_len = 50
#         padded_latents = []
#         for i in range(valid_mask.shape[0]):
#             curr_latents = image_latents[valid_idx[0] == i]
#             pad_len = max_len - len(curr_latents)
#             if pad_len > 0:
#                 padding = torch.zeros(pad_len, curr_latents.shape[-1], device=curr_latents.device)
#                 curr_latents = torch.cat([curr_latents, padding], dim=0)
#             padded_latents.append(curr_latents)
#         image_latents = torch.stack(padded_latents)
        
#         # Calculate contrastive loss
#         pos_sim = torch.einsum('bd,bmd->bm', image_latents_mean.detach(), text_latents)  # shape: (batch_size, num_features)
#         neg_sim = torch.einsum('bd,bmd->bm', image_latents_mean.detach(), neg_text_latents)  # shape: (batch_size, num_features)
#         logits = torch.stack([pos_sim, neg_sim], dim=2) / self.temperature  # shape: (batch_size, num_features, 2)
#         labels = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=logits.device)  # shape: (batch_size, num_features)
#         self.contrastive_loss = nn.CrossEntropyLoss()(
#             logits.view(-1, 2),  # reshape to (batch_size * num_features, 2)
#             labels.view(-1)      # reshape to (batch_size * num_features)
#         )

#         # Calculate orthogonal loss
#         self.orthogonal_loss = orthogonal_loss(torch.cat([text_latents, neg_text_latents], dim=1).transpose(1, 2))

#         # Feature pooling with attention
#         pooled_image_latents = []
#         pooled_text_latents = []
#         for i in range(valid_mask.shape[0]):
#             valid_features = image_latents[i, valid_mask[i] == 1]
#             attention_weights = torch.einsum('md,ld->ml', text_latents[i], valid_features)
#             attention_image = attention_weights.mean(dim=0)
#             attention_text = attention_weights.mean(dim=-1)
#             mean_image_features = torch.einsum('l,ld->d', attention_image, valid_features)
#             mean_image_features = mean_image_features / mean_image_features.norm(dim=-1, keepdim=True)
#             mean_text_features = torch.einsum('m,md->d', attention_text, text_latents[i])
#             mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
#             pooled_image_latents.append(mean_image_features)
#             pooled_text_latents.append(mean_text_features)
        
#         image_latents = torch.stack(pooled_image_latents)  # Shape: (batch, channels)
#         text_latents = torch.stack(pooled_text_latents)  # Shape: (batch, channels)
        
#         # Combine features and classify
#         combined_features = torch.cat([image_latents, text_latents], dim=1)
#         latents = self.fusion(combined_features)
#         x = self.classifier(latents)
        
#         return x


class Model(nn.Module):
    def __init__(self, encoder, ori_encoder, text_prompt_len, 
                 num_slices, num_time_points, num_co_attention_layers, d_model, num_heads, d_ffn, dropout,
                 num_agg_heads, num_agg_layers_s1, num_agg_layers_s2):
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

        self.classifier = nn.Linear(512*2, 2)
        self.dropout = nn.Dropout(0.1)  # Add dropout layer
        self.relu = nn.ReLU()
        self.norm_image = nn.BatchNorm1d(512*2)
        self.norm_text = nn.BatchNorm1d(512)

        # 存储维度信息
        self.S = num_slices        # S, e.g., 5
        self.T = num_time_points   # T, e.g., 3

        self.text_encoder = TextEncoder(self.ori_encoder)
        self.prompt_learner = nn.ModuleList([PromptLearner(Prompts[i], self.ori_encoder, text_prompt_len, 0, 0, False) for i in range(len(Prompts))])
        self.tokenized_prompts = [learner.tokenized_prompts for learner in self.prompt_learner]
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # 1. 为图像的3个时期（T=3）定义时序位置编码
        #    维度: (1, 1, T, D)，为了能与 (B, S, T, D) 的图像特征广播相加
        self.image_time_pos_encoder = nn.Parameter(torch.randn(1, 1, num_time_points, d_model))

        # 2. 为文本的5个切片（S=5）定义空间位置编码
        #    维度: (1, S, 1, D)，为了能与 (B, S, Q, D) 的文本特征广播相加
        self.text_slice_pos_encoder = nn.Parameter(torch.randn(1, num_slices, 1, d_model))

        # --- 2. 协同注意力模型初始化 ---
        self.co_attention_model = CoAttentionModel(
            num_co_attention_layers=num_co_attention_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ffn=d_ffn,
            dropout=dropout
        )

        self.aggregator = AggregationTransformer(
            d_model=d_model,
            num_heads=num_agg_heads,
            d_ffn=d_ffn, # 可以复用 d_ffn
            dropout=dropout,
            num_layers_stage1=num_agg_layers_s1,
            num_layers_stage2=num_agg_layers_s2
        )

    def forward(self, inputs, rad_feat):
        # print(f"inputs.shape: {inputs.shape}") (Batch, Channel, H, W, Total slices)
        B, C, H, W, TotalSlices = inputs.shape
        # print(f"rad_feat.shape: {rad_feat.shape}") (Batch, slice, questions)
        _, S_rad, Q_rad = rad_feat.shape

        # 安全检查：确保输入的总切片数与模型初始化的S和T匹配
        assert TotalSlices == self.S * self.T, \
            f"Input tensor's last dimension ({TotalSlices}) does not match S*T ({self.S * self.T})"
        assert S_rad == self.S, \
            f"rad_feat's slice dimension ({S_rad}) does not match model's S ({self.S})"


        # Part 1: 细粒度特征提取 (现在是动态的)
        # --- 图像特征提取 --- (B, C, H, W, TotalSlice) -> (B, TotalSlice, C, H, W)
        x_permuted = inputs.permute(0, 4, 1, 2, 3)
        x_reshaped = x_permuted.reshape(B * TotalSlices, C, H, W)
        image_latents_flat = self.encoder.encode_image(x_reshaped).float()
        image_latents = image_latents_flat.reshape(B, self.S, self.T, -1)
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
        image_latents = image_latents + self.image_time_pos_encoder # positional embedding

        # --- 文本特征提取 ---
        prompts = []
        for i in range(len(self.prompt_learner)):
            prompts.append(self.prompt_learner[i]())
        tokenized_prompts = self.tokenized_prompts
        # (9, 2, d)
        self.text_feats = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(prompts, tokenized_prompts)]

        feat_indices = rad_feat.long()  # Convert to integer indices (B, S, Q)
        text_latents = torch.stack([  # 遍历 Batch (i)
            torch.stack([  # 新增: 遍历切片 (s)
                torch.stack([  # 遍历问题 (j)
                    self.text_feats[j][feat_indices[i, s, j]] # 索引方式从 feat_indices[i, j] 变为 feat_indices[i, s, j]
                    for j in range(rad_feat.shape[2]) # Q
                ])
                for s in range(rad_feat.shape[1]) # S
            ])
            for i in range(rad_feat.shape[0]) # B
        ])
        text_latents = text_latents + self.text_slice_pos_encoder # positional embedding

        # Part 2: 协同注意力融合
        # print(f'image_latents.shape: {image_latents.shape}') # torch.Size([B, 5, 3, d])
        # print(f'text_feats.shape: {text_latents.shape}')    # torch.Size([B, 5, 9, d])
        fused_v, fused_t = self.co_attention_model(image_latents, text_latents)
        # print(f'fused_v.shape: {fused_v.shape}') # (B, 5, 3, D)
        # print(f'fused_t.shape: {fused_t.shape}') # (B, 5, 9, D)

        # Part 3: 特征聚合与分类 
        output = self.aggregator(fused_t)
        
        return output


# ==============================================================================
# 模块1: CoAttentionLayer 
# ==============================================================================
class CoAttentionLayer(nn.Module):
    """
    一个完整的协同注意力层。(此模块与之前完全相同，无需修改num_co_attention_layers)
    """
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        
        # 文本到图像的注意力 (T2I)
        self.t2i_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        # FFN
        self.t2i_ffn = nn.Sequential( 
            nn.Linear(d_model, d_ffn), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_ffn, d_model)
        )
        self.t2i_norm1 = nn.LayerNorm(d_model)
        self.t2i_norm2 = nn.LayerNorm(d_model)
        self.t2i_dropout = nn.Dropout(dropout)
        
        # 图像到文本的注意力 (I2T)
        self.i2t_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.i2t_ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_ffn, d_model)
        )
        self.i2t_norm1 = nn.LayerNorm(d_model)
        self.i2t_norm2 = nn.LayerNorm(d_model)
        self.i2t_dropout = nn.Dropout(dropout)

    def forward(self, s_v: torch.Tensor, s_t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # T2I-Attention
        t_updated, _ = self.t2i_attention(query=s_t, key=s_v, value=s_v)
        s_t = self.t2i_norm1(s_t + self.t2i_dropout(t_updated))
        t_ffn_updated = self.t2i_ffn(s_t)
        s_t = self.t2i_norm2(s_t + self.t2i_dropout(t_ffn_updated))
        
        # I2T-Attention
        v_updated, _ = self.i2t_attention(query=s_v, key=s_t, value=s_t)
        s_v = self.i2t_norm1(s_v + self.i2t_dropout(v_updated))
        v_ffn_updated = self.i2t_ffn(s_v)
        s_v = self.i2t_norm2(s_v + self.i2t_dropout(v_ffn_updated))
        
        return s_v, s_t

class CoAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float):
        super().__init__()

        # 文本到图像的注意力 T2I
        self.t2i_norm_attn = nn.LayerNorm(d_model)
        self.t2i_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.t2i_dropout_attn = nn.Dropout(dropout)

        self.t2i_norm_ffn = nn.LayerNorm(d_model)
        self.t2i_ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_ffn, d_model)
        )
        self.t2i_dropout_ffn = nn.Dropout(dropout)

        # 图像到文本的注意力 I2T 
        self.i2t_norm_attn = nn.LayerNorm(d_model)
        self.i2t_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.i2t_dropout_attn = nn.Dropout(dropout)

        self.i2t_norm_ffn = nn.LayerNorm(d_model)
        self.i2t_ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_ffn, d_model)
        )
        self.i2t_dropout_ffn = nn.Dropout(dropout)

    def forward(self, s_v: torch.Tensor, s_t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # --- T2I-Attention (Pre-Norm) ---
        s_t_norm = self.t2i_norm_attn(s_t)
        s_v_norm_for_t2i = self.i2t_norm_attn(s_v) # 复用I2T的第一个norm
        t_updated, _ = self.t2i_attention(query=s_t_norm, key=s_v_norm_for_t2i, value=s_v_norm_for_t2i)
        s_t = s_t + self.t2i_dropout_attn(t_updated)

        s_t_norm = self.t2i_norm_ffn(s_t)
        t_ffn_updated = self.t2i_ffn(s_t_norm)
        s_t = s_t + self.t2i_dropout_ffn(t_ffn_updated)

        # --- I2T-Attention (Pre-Norm) ---
        s_v_norm = self.i2t_norm_attn(s_v)
        s_t_norm_for_i2t = self.t2i_norm_attn(s_t) # 复用T2I的第二个norm
        v_updated, _ = self.i2t_attention(query=s_v_norm, key=s_t_norm_for_i2t, value=s_t_norm_for_i2t)
        s_v = s_v + self.i2t_dropout_attn(v_updated)

        s_v_norm = self.i2t_norm_ffn(s_v)
        v_ffn_updated = self.i2t_ffn(s_v_norm)
        s_v = s_v + self.i2t_dropout_ffn(v_ffn_updated)

        return s_v, s_t

# ==============================================================================
# 模块2: CoAttentionModel 
# ==============================================================================
class CoAttentionModel(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float, num_co_attention_layers=4):
        super().__init__()
        
        # 直接构建 N 个协同注意力层
        self.co_attention_layers = nn.ModuleList(
            [CoAttentionLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_co_attention_layers)]
        )
        
    def forward(self, raw_image_features: torch.Tensor, raw_text_features: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        执行细粒度的协同注意力融合。

        输入:
        raw_image_features (torch.Tensor): 初始图像特征, 维度 (B, 5, 3, D_model) B=Batch, 5=切片数, 3=时期数
        raw_text_features (torch.Tensor): 初始文本特征, 维度 (B, 5, 9, D_model) B=Batch, 5=切片数, 9=问题数
        """
        
        # --- 1. 获取维度信息 ---
        B, S, T, D_img = raw_image_features.shape
        _, _, Q, D_txt = raw_text_features.shape
        # B=Batch, S=5, T=3, Q=9

        # --- 2. 向量化：将 Batch 和 切片 维度合并 ---
        # 创建一个 (B * 5) 大小的“伪批次”，以便一次性处理所有5个切片的注意力计算
        # (B, 5, 3, D) -> (B * 5, 3, D)
        s_v = raw_image_features.view(B * S, T, D_img)
        
        # (B, 5, 9, D) -> (B * 5, 9, D)
        s_t = raw_text_features.view(B * S, Q, D_txt)
        
        # --- 3. 依次通过 N 个协同注意力层 ---
        # 这里的 s_v 和 s_t 已经是被“压平”后的大批次数据
        # 注意力计算还是在每个切片的内部进行
        for layer in self.co_attention_layers:
            s_v, s_t = layer(s_v, s_t)
            
        # --- 4. 恢复原始的 Batch 和 切片 维度 ---
        # (B * 5, 3, D) -> (B, 5, 3, D)
        final_s_v = s_v.view(B, S, T, D_img)
        
        # (B * 5, 9, D) -> (B, 5, 9, D)
        final_s_t = s_t.view(B, S, Q, D_txt)
        
        return final_s_v, final_s_t


# ==============================================================================
# 模块3: AggregationTransformer 
# ==============================================================================
class AggregationTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float, 
                 num_layers_stage1: int, num_layers_stage2: int):
        super().__init__()
        
        self.d_model = d_model

        # --- Stage 1: Per-Question Aggregation (across 5 slices) ---
        # 可学习的 [CLS] Token，用于聚合5个切片的特征
        self.cls_token_stage1 = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Stage 1 的 Transformer Encoder (9, 5, d) -> (9, d)
        stage1_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ffn, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_stage1 = nn.TransformerEncoder(
            stage1_encoder_layer, num_layers=num_layers_stage1
        )

        # --- Stage 2: Final Aggregation (across 9 questions) ---
        # 另一个【独立】的可学习 [CLS] Token，用于聚合9个问题的特征
        self.cls_token_stage2 = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Stage 2 的 Transformer Encoder (9, d)-> (d)
        stage2_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ffn, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_stage2 = nn.TransformerEncoder(
            stage2_encoder_layer, num_layers=num_layers_stage2
        )

        # --- Stage 3: Classification Head ---
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model), # LayerNorm 通常比 BatchNorm 更适合Transformer的输出
            nn.GELU(),
            nn.Dropout(dropout),
            # 最终的MLP，从d_model维降到2维
            nn.Linear(d_model, 2)
        )

    def forward(self, fused_t: torch.Tensor) -> torch.Tensor:
        B, S, Q, D = fused_t.shape # (B, 5, 9, D)
        
        # --- Stage 1: Per-Question Aggregation ---
        # 1. 重排维度，将“问题”作为“伪批次”的一部分，以便并行处理
        # (B, 5, 9, D) -> (B, 9, 5, D)
        t_permuted = fused_t.permute(0, 2, 1, 3)
        # (B, 9, 5, D) -> (B * 9, 5, D)
        t_reshaped = t_permuted.reshape(B * Q, S, D)

        # 2. 为每个序列添加 [CLS] Token
        # 扩展 [CLS] Token 以匹配“伪批次”的大小
        cls_tokens_s1 = self.cls_token_stage1.expand(B * Q, -1, -1)
        # 拼接在序列开头
        stage1_input = torch.cat([cls_tokens_s1, t_reshaped], dim=1) # (B * 9, 5+1, D)

        # 3. 通过 Stage 1 Transformer
        stage1_output = self.transformer_stage1(stage1_input)

        # 4. 提取每个序列的 [CLS] Token 输出作为聚合后的特征
        aggregated_per_question = stage1_output[:, 0, :] # (B * 9, D)

        # --- Stage 2: Final Aggregation ---
        # 5. 恢复维度，得到所有问题的聚合向量
        # (B * 9, D) -> (B, 9, D)
        stage2_input_features = aggregated_per_question.view(B, Q, D)

        # 6. 再次添加 [CLS] Token
        cls_tokens_s2 = self.cls_token_stage2.expand(B, -1, -1)
        stage2_input = torch.cat([cls_tokens_s2, stage2_input_features], dim=1) # (B, 9+1, D)

        # 7. 通过 Stage 2 Transformer
        stage2_output = self.transformer_stage2(stage2_input)
        
        # 8. 提取最终的 [CLS] Token 输出，作为整个病人的特征表示
        patient_feature = stage2_output[:, 0, :] # (B, D)

        # --- Stage 3: Classification ---
        # 9. 将最终特征送入分类头
        logits = self.classification_head(patient_feature) # (B, 2)
        
        return logits







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
