from einops import rearrange
from math import log
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.torch as ptu
from model.utils import padding, unpadding 
from patch.patchinfo import CenterPoint, CenterPointSet
from timm.models.layers import trunc_normal_


def recover_SAPS(scores, dstbt, im_size, min_patchsize):
    
    def lineInterpolate(srcCoords, srcPoints, dstCoord, dstLen):
        if srcCoords[0][0] == srcCoords[1][0]:    # The two source points have the same y coordinate
            if dstCoord[1] > srcCoords[1][1] or dstCoord[1] < srcCoords[0][1]:
                raise ValueError("The destination point must be between the two source points.")
            else:
                k1 = (srcCoords[1][1] - dstCoord[1]) / (srcCoords[1][1] - srcCoords[0][1])
                k2 = (dstCoord[1] - srcCoords[0][1]) / (srcCoords[1][1] - srcCoords[0][1])
                
        elif srcCoords[0][1] == srcCoords[1][1]:    # The two source points have the same x coordinate
            if dstCoord[0] > srcCoords[1][0] or dstCoord[0] < srcCoords[0][0]:
                raise ValueError("The destination point must be between the two source points.")
            else:
                k1 = (srcCoords[1][0] - dstCoord[0]) / (srcCoords[1][0] - srcCoords[0][0])
                k2 = (dstCoord[0] - srcCoords[0][0]) / (srcCoords[1][0] - srcCoords[0][0])
        else:
            raise ValueError("The two source points must be in a horizontal or a vertical line.")
        
        tgt_fml = np.zeros([dstLen], dtype='float32')
        tgt_fml = tgt_fml + srcPoints[0].formula() * k1 + srcPoints[1].formula() * k2
            
        return tgt_fml
    
    def squareInterpolate(srcCoords, srcPoints, dstCoord, dstLen):
        if dstCoord[0] > srcCoords[2][0] or dstCoord[0] < srcCoords[0][0] or dstCoord[1] > srcCoords[1][1] or dstCoord[1] < srcCoords[0][1]:
            raise ValueError("The destination point must be inside the square area defined by the four source points.")
        
        a1 = srcCoords[2][0] - dstCoord[0]
        a2 = dstCoord[0] - srcCoords[0][0]
        b1 = srcCoords[1][1] - dstCoord[1]
        b2 = dstCoord[1] - srcCoords[0][1]
        a3b3 = (srcCoords[2][0] - srcCoords[0][0]) * (srcCoords[1][1] - srcCoords[0][1])
        k1 = a1 * b1 / a3b3
        k2 = a1 * b2 / a3b3
        k3 = a2 * b1 / a3b3
        k4 = a2 * b2 / a3b3
        
        tgt_fml = np.zeros([dstLen], dtype='float32')
        tgt_fml = tgt_fml + srcPoints[0].formula() * k1 + srcPoints[1].formula() * k2 + srcPoints[2].formula() * k3 + srcPoints[3].formula() * k4
            
        return tgt_fml
    
    H, W = im_size
    B = len(scores)
    
    masks = torch.Tensor().to(ptu.device)
    
    for i in range(B):
        num_patches = scores[i].size(1)
        
        # First, we have to find out how many different kinds of patch size there are in an image, 
        # that is the number of CenterPointSet objects we need.
        min_size = np.min(dstbt[i][:, 2])
        max_size = np.max(dstbt[i][:, 2])
        # The formulas used to compute the final classes scores
        H_minpatch = H//min_size
        W_minpatch = W//min_size
        formulas = np.zeros([H_minpatch, W_minpatch, num_patches], dtype="float32")
        rec = np.zeros([H_minpatch, W_minpatch], dtype='int8')
        
        # Fill the CenterPointSet objects into a list.
        cpset_list = list()
        j = min_size
        while j <= max_size:
            cpset_list.append(CenterPointSet(j))
            j = j * 2
        
        # Add the known center points into the center point sets.
        processStack = list()
        for j in range(num_patches):
            patch_size = dstbt[i][j, 2]
            patch_r = dstbt[i][j, 3]
            patch_c = dstbt[i][j, 4]
            set_index = int(log(patch_size/min_size, 2))
            ctr_fml = np.zeros([num_patches], dtype='float32')
            ctr_fml[j] = 1
            cpset_list[set_index].add(CenterPoint(patch_r, patch_c, patch_size, ctr_fml, True, ori_index=j))
            if patch_size == min_size:
                formulas[patch_r//min_size, patch_c//min_size] = ctr_fml
                rec[patch_r//min_size, patch_c//min_size] = 1
            # Interpolate some critical center points and add them into the center point sets.
            while True:
                if len(processStack) > 0:
                    top = processStack[-1]
                    if patch_size == top[0]:
                        if top[1] == 3 and patch_size != max_size:          # 4 patches are all ready  
                            patch_size = patch_size * 2
                            patch_r = processStack[-3][2]
                            patch_c = processStack[-3][3]
                            processStack = processStack[:-3]
                            
                            ctr_fml = np.zeros([num_patches], dtype='float32')
                            ctr_fml = ctr_fml + (cpset_list[set_index][-4].formula() + 
                                                 cpset_list[set_index][-3].formula() + 
                                                 cpset_list[set_index][-2].formula() + 
                                                 cpset_list[set_index][-1].formula()) * 0.25
                            
                            set_index = int(log(patch_size/min_size, 2))
                            cpset_list[set_index].add(CenterPoint(patch_r, patch_c, patch_size, ctr_fml, False))
                            continue
                        else:
                            processStack.append((patch_size, top[1]+1, patch_r, patch_c))
                            break
                    else:
                        processStack.append((patch_size, 1, patch_r, patch_c))
                        break
                else:
                    processStack.append((patch_size, 1, patch_r, patch_c))
                    break
        
        # Complete the center points from the largest patch size till the smallest.
        for j in range(len(cpset_list)-1, 0, -1):
            cpset = cpset_list[j]
            cpset_to = cpset_list[j-1]
            cpset.setAreaPoint((H, W))
            for cpoint in cpset:
                cp_y = cpoint.coord()[0]
                cp_x = cpoint.coord()[1]
                cp_size = cpoint.size()
                # First, we interpolate the center points in inner areas
                if cpoint.hasInnerArea():
                    srcCoords = [cpoint.coord(), (cp_y, cp_x+cp_size), (cp_y+cp_size, cp_x), (cp_y+cp_size, cp_x+cp_size)]
                    srcPoints = [cpoint, cpset.findPoint(srcCoords[1]), cpset.findPoint(srcCoords[2]), cpset.findPoint(srcCoords[3])]
                    tgt_y = cp_y + cp_size/4
                    tgt_x = cp_x + cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y)
                    tgt_c = int(cp_x)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = squareInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    tgt_x = cp_x + 3*cp_size/4         # The target point's coordinate
                    tgt_c = int(cp_x + cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = squareInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    tgt_y = cp_y + 3*cp_size/4
                    tgt_x = cp_x + cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y + cp_size/2)
                    tgt_c = int(cp_x)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = squareInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    tgt_x = cp_x + 3*cp_size/4         # The target point's coordinate
                    tgt_c = int(cp_x + cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = squareInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    
                if cpoint.hasTopArea():
                    srcCoords = [cpoint.coord(), (cp_y, cp_x+cp_size)]
                    srcPoints = np.array([cpoint, cpset.findPoint(srcCoords[1])])
                    tgt_y = cp_y - cp_size/4
                    tgt_x = cp_x + cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y - cp_size/2)
                    tgt_c = int(cp_x)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    tgt_x = cp_x + 3*cp_size/4         # The target point's coordinate
                    tgt_c = int(cp_x + cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                
                if cpoint.hasBottomArea():
                    srcCoords = [cpoint.coord(), (cp_y, cp_x+cp_size)]
                    srcPoints = np.array([cpoint, cpset.findPoint(srcCoords[1])])
                    tgt_y = cp_y + cp_size/4
                    tgt_x = cp_x + cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y)
                    tgt_c = int(cp_x)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    tgt_x = cp_x + 3*cp_size/4         # The target point's coordinate
                    tgt_c = int(cp_x + cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                            
                if cpoint.hasLeftArea():
                    srcCoords = [cpoint.coord(), (cp_y+cp_size, cp_x)]
                    srcPoints = np.array([cpoint, cpset.findPoint(srcCoords[1])])
                    tgt_y = cp_y + cp_size/4
                    tgt_x = cp_x - cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y)
                    tgt_c = int(cp_x - cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    tgt_y = cp_y + 3*cp_size/4
                    tgt_r = int(cp_y + cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                
                if cpoint.hasRightArea():
                    srcCoords = [cpoint.coord(), (cp_y+cp_size, cp_x)]
                    srcPoints = np.array([cpoint, cpset.findPoint(srcCoords[1])])
                    tgt_y = cp_y + cp_size/4
                    tgt_x = cp_x + cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y)
                    tgt_c = int(cp_x)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                    tgt_y = cp_y + 3*cp_size/4
                    tgt_r = int(cp_y + cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = lineInterpolate(srcCoords, srcPoints, (tgt_y, tgt_x), num_patches)
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1 
                
                if cpoint.hasTopLeftArea():
                    tgt_y = cp_y - cp_size/4
                    tgt_x = cp_x - cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y - cp_size/2)
                    tgt_c = int(cp_x - cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = cpoint.formula()
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                
                if cpoint.hasTopRightArea():
                    tgt_y = cp_y - cp_size/4
                    tgt_x = cp_x + cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y - cp_size/2)
                    tgt_c = int(cp_x)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = cpoint.formula()
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                
                if cpoint.hasBottomLeftArea():
                    tgt_y = cp_y + cp_size/4
                    tgt_x = cp_x - cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y)
                    tgt_c = int(cp_x - cp_size/2)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = cpoint.formula()
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
                
                if cpoint.hasBottomRightArea():
                    tgt_y = cp_y + cp_size/4
                    tgt_x = cp_x + cp_size/4         # The target point's coordinate
                    tgt_r = int(cp_y)
                    tgt_c = int(cp_x)
                    if cpset_to.findPoint((tgt_y, tgt_x)) == None:
                        ctr_fml = cpoint.formula()
                        cpset_to.add(CenterPoint(tgt_r, tgt_c, cpset_to.size(), ctr_fml, False))
                        if j == 1:
                            formulas[tgt_r//min_size, tgt_c//min_size] = ctr_fml
                            rec[tgt_r//min_size, tgt_c//min_size] = 1
        
        if 0 in rec:
            raise ValueError("There shouldn't be any unfilled pixels. There must be some mistakes.")
        
        # Our segmentation image, come back!
        formulas = formulas.reshape([H_minpatch*W_minpatch, num_patches])
        mask = torch.tensor(formulas).to(ptu.device)
        mask = mask @ scores[i][0]
        mask = mask.reshape([H_minpatch, W_minpatch, -1]).unsqueeze(0)
        masks = torch.cat([masks, mask])
    
    masks = rearrange(masks, "b h w n -> b n h w")
    return masks
    
    
class Segmenter_SAPS(nn.Module):
    def __init__(self, encoder, decoder, n_cls):
        super().__init__()
        self.n_cls = n_cls
        self.min_patchsize = encoder.min_patchsize
        self.encoder = encoder
        self.decoder = decoder
        
    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params
    
    def forward(self, im, im_edge):
        H_ori, W_ori = im.size(2), im.size(3)
        basicFactor = self.min_patchsize * 16
        im = padding(im, basicFactor)
        H, W = im.size(2), im.size(3)

        x, dstbt = self.encoder(im, im_edge, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        for i in range(len(x)):
            x[i] = x[i][:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))
        masks = recover_SAPS(masks, dstbt, (H, W), self.min_patchsize)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks
    
    def get_attention_map_enc(self, im, im_edge, layer_id):
        return self.encoder.get_attention_map(im, im_edge, layer_id)
    
    def get_attention_map_dec(self, im, im_edge, layer_id):
        x, dstbt = self.encoder(im, im_edge, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        for i in range(len(x)):
            x[i] = x[i][:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
