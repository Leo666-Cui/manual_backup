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

        self.classifier = nn.Linear(512*2, 2)
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
        x = self.classifier(latents)
        
        return x

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
