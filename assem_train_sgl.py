from datetime import datetime

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import visdom

import sapsconfig as cfg
import config

import utils.torch as ptu
from model.factory import create_segmenter_SAPS
from optim.factory import lr_scheduler_polynomial

from rgblabel import onehot2mbatchrgb, class2mbatchrgb, onehot2mbatchclass
from confmateval import mbatch2confmats, mbatchiou_via_confmats


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


# Parameters backbone and decoder, choose from config.yml
def train(backbone, decoder, dataset="assembly", dropout=0.0, drop_path=0.1, load_checkpoint=0):
    # start distributed mode
    ptu.set_gpu_mode(use=True)

    vis = visdom.Visdom()
    
    # set up configuration
    cfgyml = config.load_config()
    model_cfg = cfgyml["model"][backbone]
    decoder_cfg = cfgyml["decoder"][decoder]
    dataset_cfg = cfgyml["dataset"][dataset]
    
    # model config
    im_size = dataset_cfg["im_size"]
    crop_size = dataset_cfg.get("crop_size", im_size)
    window_size = dataset_cfg.get("window_size", im_size)
    window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg
    model_cfg["div_thres"] = cfg.div_thres
    model_cfg["min_patchsize"] = cfg.min_patchsize
    
    # dataset config
    batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    base_lr = dataset_cfg["learning_rate"]
    
    if cfg.dataset == 2:
        from AssemData3D import train_dataset_2
        from AssemData3D import val_dataset_2
        train_dataloader = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, num_workers=4, 
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=True, num_workers=4, 
                                    pin_memory=True)
    else:
        from AssemData3D import train_dataset_1
        from AssemData3D import val_dataset_1
        train_dataloader = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, num_workers=4, 
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=True, num_workers=4, 
                                    pin_memory=True) 
    
    # model
    net_kwargs = model_cfg
    net_kwargs["n_cls"] = cfg.classes
    segmenter_model = create_segmenter_SAPS(net_kwargs)
    segmenter_model = segmenter_model.to(ptu.device)
    n_parameters = sum(p.numel() for p in segmenter_model.parameters() if p.requires_grad)
    print("%d parameters in this model." %(n_parameters))
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(ptu.device)
    
    # optimizer
    optimizer = optim.SGD(segmenter_model.parameters(), lr=base_lr, momentum=cfg.momentum, 
                          weight_decay=cfg.weight_decay)
    iter_max = len(train_dataloader) * num_epochs
    scheduler = lr_scheduler_polynomial(optimizer, 
                                        poly_step_size=1, 
                                        iter_warmup=0.0, 
                                        iter_max=iter_max, 
                                        poly_power=0.9, 
                                        min_lr=1e-05)
    
    start_epo = 0
    
    all_train_epoch_loss = []
    all_val_epoch_loss = []
    all_train_epoch_miou = []
    all_val_epoch_miou = []
    
    if not load_checkpoint == 0:
        checkpoint = torch.load('checkpoints/vitsaps_{}_model_dataset{}_{}.pt'.format(cfg.model, cfg.dataset, 
                                                                                      load_checkpoint))
        start_epo = checkpoint["epoch_num"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        segmenter_model.load_state_dict(checkpoint["model"])
        all_train_epoch_loss = checkpoint["train_epoch_loss"]
        all_val_epoch_loss = checkpoint["val_epoch_loss"]
        all_train_epoch_miou = checkpoint["train_epoch_miou"]
        all_val_epoch_miou = checkpoint["val_epoch_miou"]
    

    # start timing
    prev_time = datetime.now()
    for epo in range(start_epo, num_epochs):
        
        train_loss = 0
        train_iou = np.zeros([1, cfg.classes])
        segmenter_model.train()
        for index, (assem, assem_edge, assem_msk) in enumerate(train_dataloader):
            
            # assem.shape is torch.Size([4, 3, 384, 384])
            # assem_msk.shape is torch.Size([4, 384, 384])
            assem = assem.to(ptu.device)
            assem_msk = assem_msk.long()
            assem_msk = assem_msk.to(ptu.device)

            # output.shape is torch.Size([4, 17, 384, 384])
            output = segmenter_model(assem, assem_edge)
            loss = criterion(output, assem_msk)

            optimizer.zero_grad()
            loss.backward()
            iter_loss = loss.item()
            #all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()
            scheduler.step()

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 17, 384, 384)
            output_rgb = onehot2mbatchrgb(output_np)
            output_cls = onehot2mbatchclass(output_np)
            assem_msk_np = assem_msk.cpu().detach().numpy().copy() # assem_msk_np.shape = (4, 384, 384) 
            assem_msk_rgb = class2mbatchrgb(assem_msk_np)
            
            # Get IoU of the minibatch via confusion matrices
            confmats = mbatch2confmats(cfg.classes, output_cls, assem_msk_np)
            iter_iou = mbatchiou_via_confmats(confmats)
            train_iou = np.concatenate([train_iou, iter_iou], axis=0)

            if np.mod(index+1, 50) == 0:
                print('epoch {}, {}/{}, training loss is {}'.format(epo+1, index+1, len(train_dataloader), iter_loss))
                # vis.close()
                vis.images(output_rgb, win='train_pred', opts=dict(title='train prediction')) 
                vis.images(assem_msk_rgb, win='train_label', opts=dict(title='label'))
        train_iou = np.delete(train_iou, 0, axis=0)

        
        val_loss = 0
        val_iou = np.zeros([1, cfg.classes])
        segmenter_model.eval()
        with torch.no_grad():
            print(r'Validating... Open http://localhost:8097/ to see validation result.')
            
            for index, (assem, assem_edge, assem_msk) in enumerate(val_dataloader):

                assem = assem.to(ptu.device)
                assem_msk = assem_msk.long()
                assem_msk = assem_msk.to(ptu.device)

                output = segmenter_model(assem, assem_edge)    # output.shape is torch.Size([4, 1, 384, 384])
                loss = criterion(output, assem_msk)
                iter_loss = loss.item()
                #all_val_iter_loss.append(iter_loss)
                val_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 17, 384, 384)
                output_rgb = onehot2mbatchrgb(output_np)
                output_cls = onehot2mbatchclass(output_np)
                assem_msk_np = assem_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 384, 384) 
                assem_msk_rgb = class2mbatchrgb(assem_msk_np)
                
                # Get IoU of the minibatch via confusion matrices
                confmats = mbatch2confmats(cfg.classes, output_cls, assem_msk_np)
                iter_iou = mbatchiou_via_confmats(confmats)
                val_iou = np.concatenate([val_iou, iter_iou], axis=0)
        
                if np.mod(index+1, 50) == 0:
                    print('epoch {}, {}/{}, validation loss is {}'.format(epo+1, index+1, len(val_dataloader), 
                                                                          iter_loss))
                    # vis.close()
                    vis.images(output_rgb, win='val_pred', opts=dict(title='validate prediction')) 
                    vis.images(assem_msk_rgb, win='val_label', opts=dict(title='label'))
            val_iou = np.delete(val_iou, 0, axis=0)


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        
        epoch_train_loss = train_loss/len(train_dataloader)
        epoch_val_loss = val_loss/len(val_dataloader)
        epoch_train_iou = np.nanmean(train_iou, axis=0)
        epoch_val_iou = np.nanmean(val_iou, axis=0)
        epoch_train_miou = np.nanmean(epoch_train_iou[1:])
        epoch_val_miou = np.nanmean(epoch_val_iou[1:])

        print('epoch training loss = %f, miou = %f, epoch validation loss = %f, miou = %f, %s' 
              %(epoch_train_loss, epoch_train_miou, epoch_val_loss, epoch_val_miou, time_str))
        
        all_train_epoch_loss.append(epoch_train_loss)
        all_val_epoch_loss.append(epoch_val_loss)
        all_train_epoch_miou.append(epoch_train_miou)
        all_val_epoch_miou.append(epoch_val_miou)
        # vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='training iter loss')) 
        # vis.line(all_val_iter_loss, win='val_iter_loss', opts=dict(title='validation iter loss'))
        vis.line(all_train_epoch_miou, win='train_epoch_miou',opts=dict(title='training epoch miou')) 
        vis.line(all_val_epoch_miou, win='val_epoch_miou', opts=dict(title='validation epoch miou'))
        
        max_miou = max(all_val_epoch_miou)
        print("The max validation miou is %f, at epoch %d" %(max_miou, all_val_epoch_miou.index(max_miou)+1))
        
        save_info = {"epoch_num": epo+1, 
                     "optimizer": optimizer.state_dict(), 
                     "scheduler": scheduler.state_dict(), 
                     "model": segmenter_model.state_dict(), 
                     "train_epoch_loss": all_train_epoch_loss, 
                     "val_epoch_loss": all_val_epoch_loss, 
                     "train_epoch_iou": epoch_train_iou, 
                     "val_epoch_iou": epoch_val_iou, 
                     "train_epoch_miou": all_train_epoch_miou, 
                     "val_epoch_miou": all_val_epoch_miou}
        torch.save(save_info, 'checkpoints/vitsaps_{}_model_dataset{}_{}.pt'.format(cfg.model, cfg.dataset, epo+1))
        print('saving checkpoints/vitsaps_{}_model_dataset{}_{}.pt'.format(cfg.model, cfg.dataset, epo+1))


if __name__ == "__main__":
    #torch.cuda.empty_cache()
    train(backbone=cfg.model, decoder=cfg.decoder, dropout=cfg.drop_out_rate, drop_path=cfg.drop_path_rate, 
          load_checkpoint=0)
