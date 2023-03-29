import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import visdom

import sapsconfig as cfg
import config

import utils.torch as ptu
from model.factory import create_segmenter_SAPS

from rgblabel import onehot2mbatchrgb, class2mbatchrgb, onehot2mbatchclass
from confmateval import mbatch2confmats, mbatchiou_via_confmats


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


# Parameters backbone and decoder, choose from config.yml
def test(backbone, decoder, dataset="assembly", dropout=0.0, drop_path=0.1, load_checkpoint=200):
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
    
    if cfg.dataset == 2:
        from AssemData3D import test_dataset_2
        test_dataloader = DataLoader(test_dataset_2, batch_size=batch_size, shuffle=True, num_workers=4, 
                                     pin_memory=True)
    else:
        from AssemData3D import test_dataset_1
        test_dataloader = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=True, num_workers=4, 
                                     pin_memory=True) 
    
    # model
    net_kwargs = model_cfg
    net_kwargs["n_cls"] = cfg.classes
    segmenter_model = create_segmenter_SAPS(net_kwargs)
    segmenter_model = segmenter_model.to(ptu.device)
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(ptu.device)
    
    # Load model from the checkpoint
    checkpoint = torch.load('checkpoints/vitsaps_{}_model_dataset{}_{}.pt'.format(cfg.model, cfg.dataset, 
                                                                                  load_checkpoint))
    model_epo = checkpoint["epoch_num"]
    segmenter_model.load_state_dict(checkpoint["model"])

    # start testing
    test_loss = 0
    test_iou = np.zeros([1, cfg.classes])
    segmenter_model.eval()
    with torch.no_grad():
        print(r'Testing... Open http://localhost:8097/ to see test result.')
        
        for index, (assem, assem_edge, assem_msk) in enumerate(test_dataloader):

            assem = assem.to(ptu.device)
            assem_msk = assem_msk.long()
            assem_msk = assem_msk.to(ptu.device)

            output = segmenter_model(assem, assem_edge)    # output.shape is torch.Size([4, 1, 384, 384])
            loss = criterion(output, assem_msk)
            iter_loss = loss.item()
            test_loss += iter_loss

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 17, 384, 384)
            output_rgb = onehot2mbatchrgb(output_np)
            output_cls = onehot2mbatchclass(output_np)
            assem_msk_np = assem_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 384, 384) 
            assem_msk_rgb = class2mbatchrgb(assem_msk_np)
            
            # Get IoU of the minibatch via confusion matrices
            confmats = mbatch2confmats(cfg.classes, output_cls, assem_msk_np)
            iter_iou = mbatchiou_via_confmats(confmats)
            test_iou = np.concatenate([test_iou, iter_iou], axis=0)
        
            if np.mod(index+1, 25) == 0:
                print('iteration {}/{}, test loss is {}'.format(index+1, len(test_dataloader), iter_loss))
                # vis.close()
                vis.images(output_rgb, win='test_pred', opts=dict(title='test prediction')) 
                vis.images(assem_msk_rgb, win='test_label', opts=dict(title='label'))
        test_iou = np.delete(test_iou, 0, axis=0)

    final_test_loss = test_loss/len(test_dataloader)
    final_test_iou = np.nanmean(test_iou, axis=0)
    final_test_miou = np.nanmean(final_test_iou[1:])
    
    print("Final test loss = %f" %(final_test_loss))
    print("IoU of each class:")
    for i in range(cfg.classes):
        print("Class %d: %f" %(i, final_test_iou[i]))
    print('Final test mIoU without background = %f' %(final_test_miou))
    
    save_info = {"final_test_loss": final_test_loss, 
                 "final_test_iou": final_test_iou, 
                 "final_test_miou": final_test_miou}
    torch.save(save_info, 'checkpoints/vitsaps_{}_model_dataset{}_test_result_{}.pt'.format(cfg.model, cfg.dataset, 
                                                                                            model_epo))
    print('saving checkpoints/vitsaps_{}_model_dataset{}_test_result_{}.pt'.format(cfg.model, cfg.dataset, 
                                                                                   model_epo))
        

if __name__ == "__main__":
    torch.cuda.empty_cache()
    test(backbone=cfg.model, decoder=cfg.decoder, dropout=cfg.drop_out_rate, drop_path=cfg.drop_path_rate, 
         load_checkpoint=200)
