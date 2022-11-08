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
            "F2_1" : 0,
            "Acc_1" : 0,
            "F2_2" : 0,
            "Acc_2" : 0,
            "F2_3" : 0,
            "Acc_3" : 0
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

                summ["F2_1"] += Util.F2_score(output['level_1'], img_labels['labels']['level_1'])
                summ["Acc_1"] += Util.acc_func(output['level_1'], img_labels['labels']['level_1'], batch_size)

                summ["F2_2"] += Util.F2_score(output['level_2'], img_labels['labels']['level_2'])
                summ["Acc_2"] += Util.acc_func(output['level_2'], img_labels['labels']['level_2'], batch_size)

                summ["F2_3"] += Util.F2_score(output['level_3'], img_labels['labels']['level_3'])
                summ["Acc_3"] += Util.acc_func(output['level_3'], img_labels['labels']['level_3'], batch_size)

                dict_summ = {"loss" : f'{summ["loss"]/(i+1):05.3f}'}
                dict_summ.update({"F2_1" : f'{summ["F2_1"]/(i+1):05.3f}'})
                dict_summ.update({"Acc_1" : f'{summ["Acc_1"]/(i+1)*100:05.3f}'})
                dict_summ.update({"F2_2" : f'{summ["F2_2"]/(i+1):05.3f}'})
                dict_summ.update({"Acc_2" : f'{summ["Acc_2"]/(i+1)*100:05.3f}'})
                dict_summ.update({"F2_3" : f'{summ["F2_3"]/(i+1):05.3f}'})
                dict_summ.update({"Acc_3" : f'{summ["Acc_3"]/(i+1)*100:05.3f}'})
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
            "F2_val_1" : 0,
            "F2_val_2" : 0,
            "F2_val_3" : 0,
            "Acc_val_1": 0,
            "Acc_val_2": 0,
            "Acc_val_3": 0
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
                    
                    summ["F2_val_1"] += Util.F2_score(output['level_1'], img_labels['labels']['level_1'])
                    summ["Acc_val_1"] += Util.acc_func(output['level_1'], img_labels['labels']['level_1'], batch_size)
                    summ["F2_val_2"] += Util.F2_score(output['level_2'], img_labels['labels']['level_2'])
                    summ["Acc_val_2"] += Util.acc_func(output['level_2'], img_labels['labels']['level_2'], batch_size)
                    summ["F2_val_3"] += Util.F2_score(output['level_3'], img_labels['labels']['level_3'])
                    summ["Acc_val_3"] += Util.acc_func(output['level_3'], img_labels['labels']['level_3'], batch_size)

                    dict_summ = {"loss_val" : f'{summ["loss_val"]/(i+1):05.3f}'}
                    dict_summ.update({"F2_val_1" : f'{summ["F2_val_1"]/(i+1):05.3f}'})
                    dict_summ.update({"Acc_val_1" : f'{summ["Acc_val_1"]/(i+1)*100:05.3f}'})
                    dict_summ.update({"F2_val_2" : f'{summ["F2_val_2"]/(i+1):05.3f}'})
                    dict_summ.update({"Acc_val_2" : f'{summ["Acc_val_2"]/(i+1)*100:05.3f}'})
                    dict_summ.update({"F2_val_3" : f'{summ["F2_val_3"]/(i+1):05.3f}'})
                    dict_summ.update({"Acc_val_3" : f'{summ["Acc_val_3"]/(i+1)*100:05.3f}'})
                    t.set_postfix(dict_summ)
                    t.update()
        
        performance_dict.update({key : val/(i+1) for key, val in summ.items()})
        # performance_dict.update({key : val/total_pred[key]*100 for key, val in correct_pred.items()})
        print(performance_dict)
        return performance_dict