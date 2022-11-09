import torch

from tqdm import tqdm

from util.util import Util

class Training():
    def train(dataloader, model, loss_func, optimizer, epoch, max_epoch, gpu):

        model.train()

        performance_dict = {
            "epoch": epoch+1
        }

        summ = {
            "loss": 0,
            "acc_1" : 0,
            "acc_2" : 0,
            "acc_3" : 0
        }

        with tqdm(total=len(dataloader)) as t:
            t.set_description(f'[{epoch+1}/{max_epoch}]')
            
            # Iteration step
            for i, img_labels in enumerate(dataloader):

                if gpu is not None:
                    img_labels['image'] = img_labels['image'].cuda(gpu, non_blocking=True)
                if torch.cuda.is_available():
                    for key in img_labels['labels'].keys():
                        img_labels['labels'][key] = img_labels['labels'][key].cuda(gpu, non_blocking=True)
                
                output = model(img_labels['image'])

                # Calculate Loss
                loss = Util.criterion(loss_func, output, img_labels)
                summ["loss"] += loss.item()
                
                # Train & Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_size = img_labels['labels']['level_1'].size(0)
                
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
        
        performance_dict.update({key: val/(i+1) for key, val in summ.items()})
        print(performance_dict)
        return performance_dict

    def eval(dataloader, model, loss_func, epoch, max_epoch, gpu):

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

                    output = model(img_labels['image'])
                    
                    # Calculate Loss
                    loss = Util.criterion(loss_func, output, img_labels)
                    summ["loss_val"] += loss.item()

                    # Calculate Metrics
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
        return performance_dict