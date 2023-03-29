"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import numpy as np

import torch
import torch.nn as nn

from model.utils import init_weights, resize_pos_embed
from model.blocks import Block

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.helpers import adapt_input_conv

from patch.patchinfo import PatchInfo


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x
    
class PatchEmbedding_SAPS(nn.Module):
    def __init__(self, image_size, min_patchsize, embed_dim, channels):
        super().__init__()
        self.image_size = image_size
        if image_size[0] % min_patchsize != 0 or image_size[1] % min_patchsize != 0:
            raise ValueError("Image dimensions must be divisible by the minimal patch size.")
        self.grid_size = image_size[0] // min_patchsize, image_size[1] // min_patchsize
        self.max_numpatches = self.grid_size[0] * self.grid_size[1]
        self.patchList = list()
        
        self.proj4 = nn.Conv2d(channels, embed_dim, kernel_size=4, stride=4)
        self.proj8 = nn.Conv2d(channels, embed_dim, kernel_size=8, stride=8)
        self.proj16 = nn.Conv2d(channels, embed_dim, kernel_size=16, stride=16)
        self.proj32 = nn.Conv2d(channels, embed_dim, kernel_size=32, stride=32)
        self.proj64 = nn.Conv2d(channels, embed_dim, kernel_size=64, stride=64)
        self.proj128 = nn.Conv2d(channels, embed_dim, kernel_size=128, stride=128)
        # self.proj256 = nn.Conv2d(channels, embed_dim, kernel_size=256, stride=256)
        # self.proj512 = nn.Conv2d(channels, embed_dim, kernel_size=512, stride=512)
    
    def patchListReset(self):
        self.patchList = list()
    
    def patchDivision(self, img, edge, thres, minsize, row, col):
        """Divide a patch to 4 parts if necessary.
        img: The patch, an RGB image
        edge: The edge detection image of the patch
        thres: The division threshold
        minsize: The minimal patch size of the model
        row: The row number of the patch's top left corner pixel in the original image
        col: The column number of the patch's top left corner pixel in the original image"""
        
        def recurse(img, edge, thres, minsize, row, col):
            _, img_size, _ = img.shape
            if img_size%2 == 0 and img_size//2 >= minsize:   # Be able to be further divided
                graysum = int(edge.sum())
                if graysum > thres:
                    img1 = img[:, 0:img_size//2, 0:img_size//2]
                    img2 = img[:, 0:img_size//2, img_size//2:img_size]
                    img3 = img[:, img_size//2:img_size, 0:img_size//2]
                    img4 = img[:, img_size//2:img_size, img_size//2:img_size]
                    
                    edge1 = edge[0:img_size//2, 0:img_size//2]
                    edge2 = edge[0:img_size//2, img_size//2:img_size]
                    edge3 = edge[img_size//2:img_size, 0:img_size//2]
                    edge4 = edge[img_size//2:img_size, img_size//2:img_size]
                    
                    recurse(img1, edge1, thres, minsize, row, col)
                    recurse(img2, edge2, thres, minsize, row, col+img_size//2)
                    recurse(img3, edge3, thres, minsize, row+img_size//2, col)
                    recurse(img4, edge4, thres, minsize, row+img_size//2, col+img_size//2)
                else:
                    newPatch = PatchInfo(img, self.image_size, img_size, row, col)
                    self.patchList.append(newPatch)
            else:
                newPatch = PatchInfo(img, self.image_size, img_size, row, col)
                self.patchList.append(newPatch)
                
        _, img_height, img_width = img.shape
        
        base_h = img_height
        base_w = img_width
        while base_h%2 == 0 or base_w%2 == 0:
            base_h = base_h // 2
            base_w = base_w // 2
        twos = img_height // base_h
        
        if img_height == img_width and base_h == 1 and base_w == 1:
            recurse(img, edge, thres, minsize, 0, 0)
        else:
            img_patches = list()
            edge_patches = list()
            for i in range(base_h):
                for j in range(base_w):
                    img_patches.append(img[:, i*twos:(i+1)*twos, j*twos:(j+1)*twos])
                    edge_patches.append(edge[i*twos:(i+1)*twos, j*twos:(j+1)*twos]) 
            for i in range(base_h):
                for j in range(base_w):
                    index = i*base_w + j
                    recurse(img_patches[index], edge_patches[index], thres, minsize, i*twos, j*twos)
        
        return self.patchList
    

    def forward(self, im, im_edge, div_thres, min_patchsize):
        B, C, H, W = im.shape
        batch_patch_embedding = list()
        batch_patch_dstbt = list()
        for i in range(B):
            patch_list = self.patchDivision(im[i], im_edge[i], div_thres, min_patchsize, 0, 0)
            num_patches = len(patch_list)
            if patch_list[0]._patchSize == 4:
                x = self.proj4(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            elif patch_list[0]._patchSize == 8:
                x = self.proj8(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            elif patch_list[0]._patchSize == 16:
                x = self.proj16(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            elif patch_list[0]._patchSize == 32:
                x = self.proj32(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            elif patch_list[0]._patchSize == 64:
                x = self.proj64(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            elif patch_list[0]._patchSize == 128:
                x = self.proj128(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            # elif patch_list[0]._patchSize == 256:
            #     x = self.proj256(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            # elif patch_list[0]._patchSize == 512:
            #     x = self.proj512(patch_list[0]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)
            else:
                raise ValueError("Patch size has not been registered.")
            patch_dstbt = np.array([[patch_list[0]._imgSize[0], 
                                     patch_list[0]._imgSize[1], 
                                     patch_list[0]._patchSize, 
                                     patch_list[0]._row, 
                                     patch_list[0]._col]], dtype=int)
            for j in range(1, num_patches):
                if patch_list[j]._patchSize == 4:
                    x = torch.cat([x, self.proj4(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                elif patch_list[j]._patchSize == 8:
                    x = torch.cat([x, self.proj8(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                elif patch_list[j]._patchSize == 16:
                    x = torch.cat([x, self.proj16(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                elif patch_list[j]._patchSize == 32:
                    x = torch.cat([x, self.proj32(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                elif patch_list[j]._patchSize == 64:
                    x = torch.cat([x, self.proj64(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                elif patch_list[j]._patchSize == 128:
                    x = torch.cat([x, self.proj128(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                # elif patch_list[j]._patchSize == 256:
                #     x = torch.cat([x, self.proj256(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                # elif patch_list[j]._patchSize == 512:
                #     x = torch.cat([x, self.proj512(patch_list[j]._patch.unsqueeze(0)).flatten(2).transpose(1, 2)], dim=1)
                else:
                    raise ValueError("Patch size has not been registered.")
                patch_dstbt = np.concatenate([patch_dstbt, np.array([[patch_list[j]._imgSize[0], 
                                                                      patch_list[j]._imgSize[1], 
                                                                      patch_list[j]._patchSize, 
                                                                      patch_list[j]._row, 
                                                                      patch_list[j]._col]], dtype=int)], axis=0)
            batch_patch_embedding.append(x)        # x is tensor([1, num_patches, D])
            batch_patch_dstbt.append(patch_dstbt)    # patch_dist is np.array([num_patches, 5])
            self.patchListReset()
                
        return batch_patch_embedding, batch_patch_dstbt


class ViT_SAPS(nn.Module):
    def __init__(self, image_size, patch_size, n_layers, d_model, d_ff, n_heads, div_thres, min_patchsize, n_cls, 
                 dropout=0.1, drop_path_rate=0.0, distilled=False, channels=3):
        super().__init__()
        #self.patch_embed = PatchEmbedding(image_size, patch_size, d_model, channels)
        self.patch_embed = PatchEmbedding_SAPS(image_size, min_patchsize, d_model, channels)
        self.patch_size = patch_size
        self.min_patchsize = min_patchsize
        self.div_thres = div_thres
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            #self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches+2, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.max_numpatches+2, d_model))
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            #self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches+1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.max_numpatches+1, d_model))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        """ Load weights from .npz checkpoints for official Google Brain Flax implementation
        """
        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)

        w = np.load(checkpoint_path)
        if not prefix and 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'

        if hasattr(self.patch_embed, 'backbone'):
            # hybrid
            backbone = self.patch_embed.backbone
            stem_only = not hasattr(backbone, 'stem')
            stem = backbone if stem_only else backbone.stem
            stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
            stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
            stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
            if not stem_only:
                for i, stage in enumerate(backbone.stages):
                    for j, block in enumerate(stage.blocks):
                        bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                        for r in range(3):
                            getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                            getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                            getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                        if block.downsample is not None:
                            block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                            block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                            block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
            embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
        else:
            embed_conv_w = adapt_input_conv(
                self.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
        self.patch_embed.proj.weight.copy_(embed_conv_w)
        self.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
        self.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
        min_patchsize = self.min_patchsize
        image_size = self.patch_embed.image_size
        if self.distilled:
            num_extra_tokens = 2
        else:
            num_extra_tokens = 1
        if pos_embed_w.shape != self.pos_embed.shape:
            pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w, None, (image_size[0] // min_patchsize, image_size[1] // min_patchsize), num_extra_tokens)
        self.pos_embed.copy_(pos_embed_w)
        self.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
        self.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
        if isinstance(self.head, nn.Linear) and self.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
            self.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
            self.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
        if isinstance(getattr(self.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
            self.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
            self.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
        for i, block in enumerate(self.blocks.children()):
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
            block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
            block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
            block.attn.qkv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.qkv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
            for r in range(2):
                getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
                getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
            block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
            block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

    def forward(self, im, im_edge, return_features=False):
        B, _, H, W = im.shape
        
        x, dstbt = self.patch_embed(im, im_edge, self.div_thres, self.min_patchsize)
        cls_tokens = self.cls_token
        if self.distilled:
            dist_tokens = self.dist_token
            for i in range(B):
                x[i] = torch.cat((cls_tokens, dist_tokens, x[i]), dim=1)
        else:
            for i in range(B):
                x[i] = torch.cat((cls_tokens, x[i]), dim=1)
                
        pos_embed_grid = self.pos_embed 
        num_extra_tokens = 1 + self.distilled
        W_span = W // self.min_patchsize     # The number of minimized patches each image's width equals to
        for i in range(B):         # For each single image
            num_tokens = x[i].shape[1]
            num_patches = num_tokens - num_extra_tokens
            pos_embed = pos_embed_grid[:, 0:num_extra_tokens]
            patches_dstbt = dstbt[i]
            for j in range(num_patches):        # For each patch
                # The number of minimized patches each patch's edge length equals to
                patch_span = patches_dstbt[j, 2] // self.min_patchsize
                start = (patches_dstbt[j, 3] // self.min_patchsize) * W_span + patches_dstbt[j, 4] // self.min_patchsize + num_extra_tokens
                # Collect the small patches that the current patch covers
                pos_embed_temp = pos_embed_grid[:, start:start+patch_span]
                for k in range(1, patch_span):
                    pos_embed_temp = torch.cat([pos_embed_temp, pos_embed_grid[:, start+k*W_span : start+k*W_span+patch_span]], dim=1)
                pos_embed = torch.cat([pos_embed, pos_embed_temp.mean(dim=1).unsqueeze(1)], dim=1)
            x[i] = x[i] + pos_embed     # x[i] and pos_embed are tensor([1, num_tokens, D])
            
            x[i] = self.dropout(x[i])
            
            for blk in self.blocks:
                x[i] = blk(x[i])
            x[i] = self.norm(x[i])
        
        if return_features:
            return x, dstbt
        
        for i in range(B):
            if self.distilled:
                x[i], x_dist = x[i][:, 0], x[i][:, 1]
                x[i] = self.head(x[i])
                x_dist = self.head_dist(x_dist)
                x[i] = (x[i] + x_dist) / 2
            else:
                x[i] = x[i][:, 0]
                x[i] = self.head(x[i])
        x_tensor = torch.cat(x, dim=0)
        return x_tensor, dstbt
    
    def get_attention_map(self, im, im_edge, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        
        x, dstbt = self.patch_embed(im, im_edge, self.div_thres, self.min_patchsize)
        cls_tokens = self.cls_token
        if self.distilled:
            dist_tokens = self.dist_token
            for i in range(B):
                x[i] = torch.cat((cls_tokens, dist_tokens, x[i]), dim=1)
        else:
            for i in range(B):
                x[i] = torch.cat((cls_tokens, x[i]), dim=1)
                
        pos_embed_grid = self.pos_embed 
        num_extra_tokens = 1 + self.distilled
        W_span = W // self.min_patchsize     # The number of minimized patches each image's width equals to
        for i in range(B):         # For each single image
            num_tokens = x[i].shape[1]
            num_patches = num_tokens - num_extra_tokens
            pos_embed = pos_embed_grid[:, 0:num_extra_tokens]
            patches_dstbt = dstbt[i]
            for j in range(num_patches):        # For each patch
                # The number of minimized patches each patch's edge length equals to
                patch_span = patches_dstbt[j, 2] // self.min_patchsize
                start = (patches_dstbt[j, 3] // self.min_patchsize) * W_span + patches_dstbt[j, 4] // self.min_patchsize + num_extra_tokens
                # Collect the small patches that the current patch covers
                pos_embed_temp = pos_embed_grid[:, start:start+patch_span]
                for k in range(1, patch_span):
                    pos_embed_temp = torch.cat([pos_embed_temp, pos_embed_grid[:, start+k*W_span : start+k*W_span+patch_span]], dim=1)
                pos_embed = torch.cat([pos_embed, pos_embed_temp.mean(dim=1).unsqueeze(1)], dim=1)
            x[i] = x[i] + pos_embed     # x[i] and pos_embed are tensor([1, num_tokens, D])
            
            for j, blk in enumerate(self.blocks):
                if j < layer_id:
                    x[i] = blk(x[i])
                else:
                    x[i] = blk(x[i], return_attention=True)
                    break
            
            return x, dstbt

        
