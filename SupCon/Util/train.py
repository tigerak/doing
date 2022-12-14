import numpy as np
from tqdm import tqdm

import torch

from Util.util import Util, AvgMeter

class Training():
    def con_train(dataloader, model, loss_func, optimizer, epoch, max_epoch, gpu, mode):

        model.train()

        performance_dict = {
            "epoch": epoch+1
        }

        summ = {
            "loss": 0
        }
        
        loss_meter = AvgMeter()
        
        loss = np.inf
        with tqdm(total=len(dataloader)) as t:
            t.set_description(f'[{epoch+1}/{max_epoch}]')
            
            # Iteration step
            for i, img_labels in enumerate(dataloader):
                
                images = torch.cat([img_labels['image'][0], img_labels['image'][1]], dim=0)
                
                if gpu is not None:
                    images = images.cuda(gpu, non_blocking=True)
                if torch.cuda.is_available():
                    for label_key in img_labels['labels'].keys():
                        img_labels['labels'][label_key] = img_labels['labels'][label_key].cuda(gpu, non_blocking=True)
                
                output = model.forward_con(images, mode)
                
                # Calculate Loss
                batch_size = img_labels['labels']['level_1'].size(0)
                loss = Util.criterion_con(loss_func, output, img_labels, batch_size, gpu)
                summ["loss"] += loss.item()
                
                # Train & Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_meter.update(loss.item(), batch_size)
                
                dict_summ = {"loss" : f'{summ["loss"]/(i+1):05.3f}'}
                t.set_postfix(dict_summ)
                t.update()
                
            save_file = 'D:/my_git/doing/SupCon/ckpt/con_epoch_{epoch}.pth'.format(epoch=epoch)
            Util.save_model(model, optimizer, epoch, loss, save_file)
                    
        performance_dict.update({key: val/(i+1) for key, val in summ.items()})
        print(performance_dict)
        return loss_meter, performance_dict
    
    def ce_train(dataloader, model, loss_func, optimizer, epoch, max_epoch, gpu, mode):
    
        model.train()

        performance_dict = {
            "epoch": epoch+1
        }

        summ = {
            "loss": 0,
            "acc_1": 0,
            "acc_2": 0,
            "acc_3": 0
        }
        
        loss_meter = AvgMeter()
        
        loss = np.inf
        with tqdm(total=len(dataloader)) as t:
            t.set_description(f'[{epoch+1}/{max_epoch}]')
            
            # Iteration step
            for i, img_labels in enumerate(dataloader):
                
                images = img_labels['image'][0]
                
                if gpu is not None:
                    images = images.cuda(gpu, non_blocking=True)
                if torch.cuda.is_available():
                    for label_key in img_labels['labels'].keys():
                        img_labels['labels'][label_key] = img_labels['labels'][label_key].cuda(gpu, non_blocking=True)
                
                output = model.forward_ce(images, mode)

                # Calculate Loss
                batch_size = img_labels['labels']['level_1'].size(0)
                loss = Util.criterion_ce(loss_func, output, img_labels)
                summ["loss"] += loss.item()
                
                # Train & Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_meter.update(loss.item(), batch_size)
                
                _, preds_1 = torch.max(output['level_1'].data, 1)
                correct_1 = (preds_1 == img_labels['labels']['level_1']).sum().item()
                summ["acc_1"] += correct_1 / batch_size

                _, preds_2 = torch.max(output['level_2'].data, 1)
                correct_2 = (preds_2 == img_labels['labels']['level_2']).sum().item()
                summ["acc_2"] += correct_2 / batch_size

                _, preds_3 = torch.max(output['level_3'].data, 1)
                correct_3 = (preds_3 == img_labels['labels']['level_3']).sum().item()
                summ["acc_3"] += correct_3 / batch_size


                dict_summ = {"loss" : f'{summ["loss"]/(i+1):05.3f}'}
                dict_summ.update({"acc_1" : f'{summ["acc_1"]/(i+1)*100:05.3f}'})
                dict_summ.update({"acc_2" : f'{summ["acc_2"]/(i+1)*100:05.3f}'})
                dict_summ.update({"acc_3" : f'{summ["acc_3"]/(i+1)*100:05.3f}'})
                t.set_postfix(dict_summ)
                t.update()
                
            save_file = 'D:/my_git/doing/SupCon/ckpt/ce_epoch_{epoch}.pth'.format(epoch=epoch)
            Util.save_model(model, optimizer, epoch, loss, save_file)
        
        performance_dict.update({key: val/(i+1) for key, val in summ.items()})
        print(performance_dict)
        return loss_meter, performance_dict

    def eval(dataloader, model, loss_func, epoch, max_epoch, gpu, mode):

        model.eval()
        
        performance_dict = {
            "epoch": epoch+1
        }

        summ = {
            "loss_val": 0,
            "acc_val_1": 0,
            "acc_val_2": 0,
            "acc_val_3": 0
        }
        
        loss_meter = AvgMeter()

        with tqdm(total=len(dataloader)) as t:
            with torch.no_grad():
                t.set_description(f'[{epoch+1}/{max_epoch}]')
                # Iteration step
                for i, img_labels in enumerate(dataloader):

                    if gpu is not None:
                        img_labels['image'] = img_labels['image'].cuda(gpu, non_blocking=True)
                    if torch.cuda.is_available():
                        for key in img_labels['labels'].keys():
                            img_labels['labels'][key] = img_labels['labels'][key].cuda(gpu, non_blocking=True)

                    output = model.forward_ce(img_labels['image'], mode)
                    
                    # Calculate Loss
                    batch_size = img_labels['labels']['level_1'].size(0)
                    loss = Util.criterion_ce(loss_func, output, img_labels)
                    summ["loss_val"] += loss.item()

                    # Calculate Metrics
                    loss_meter.update(loss.item(), batch_size)
                    
                    batch_size = img_labels['labels']['level_1'].size(0)
                    _, preds_1 = torch.max(output['level_1'].data, 1)
                    correct_1 = (preds_1 == img_labels['labels']['level_1']).sum().item()
                    summ["acc_val_1"] += correct_1 / batch_size

                    _, preds_2 = torch.max(output['level_2'].data, 1)
                    correct_2 = (preds_2 == img_labels['labels']['level_2']).sum().item()
                    summ["acc_val_2"] += correct_2 / batch_size

                    _, preds_3 = torch.max(output['level_3'].data, 1)
                    correct_3 = (preds_3 == img_labels['labels']['level_3']).sum().item()
                    summ["acc_val_3"] += correct_3 / batch_size

                    dict_summ = {"loss_val" : f'{summ["loss_val"]/(i+1):05.3f}'}
                    dict_summ.update({"acc_val_1" : f'{summ["acc_val_1"]/(i+1)*100:05.3f}'})
                    dict_summ.update({"acc_val_2" : f'{summ["acc_val_2"]/(i+1)*100:05.3f}'})
                    dict_summ.update({"acc_val_3" : f'{summ["acc_val_3"]/(i+1)*100:05.3f}'})
                    t.set_postfix(dict_summ)
                    t.update()
        
        performance_dict.update({key : val/(i+1) for key, val in summ.items()})
        # performance_dict.update({key : val/total_pred[key]*100 for key, val in correct_pred.items()})
        print(performance_dict)
        return loss_meter, performance_dict