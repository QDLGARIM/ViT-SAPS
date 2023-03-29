import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from model.blocks import Block, FeedForward
from model.utils import init_weights


class DecoderLinear_SAPS(nn.Module):
    def __init__(self, n_cls, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        for i in range(len(x)):
            x[i] = self.head(x[i])
        return x


class MaskTransformer_SAPS(nn.Module):
    def __init__(self, n_cls, d_encoder, n_layers, n_heads, d_model, d_ff, drop_path_rate, dropout):
        super().__init__()
        self.d_encoder = d_encoder
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        B = len(x)
        
        for i in range(B):
            x[i] = self.proj_dec(x[i])
            cls_emb = self.cls_emb.expand(x[i].size(0), -1, -1)
            x[i] = torch.cat((x[i], cls_emb), 1)
            for blk in self.blocks:
                x[i] = blk(x[i])
            x[i] = self.decoder_norm(x[i])
    
            patches, cls_seg_feat = x[i][:, : -self.n_cls], x[i][:, -self.n_cls :]
            patches = patches @ self.proj_patch
            cls_seg_feat = cls_seg_feat @ self.proj_classes
    
            patches = patches / patches.norm(dim=-1, keepdim=True)
            cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
    
            x[i] = patches @ cls_seg_feat.transpose(1, 2)
            x[i] = self.mask_norm(x[i])  

        return x

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}.")
        B = len(x)
        for i in range(B):
            x[i] = self.proj_dec(x[i])
            cls_emb = self.cls_emb.expand(x[i].size(0), -1, -1)
            x[i] = torch.cat((x[i], cls_emb), 1)
            for j, blk in enumerate(self.blocks):
                if j < layer_id:
                    x[i] = blk(x[i])
                else:
                    x[i] = blk(x[i], return_attention=True)
            return x
