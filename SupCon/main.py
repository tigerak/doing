# %%
import math
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

from PIL import Image
from PIL import ImageFile
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import timm

# %%
class CFG:
    release_mode = 'inference' # None, 'inference', 'retrain'
    DB_PATH = '/home/doinglab-hs/ak/db/eng_df.csv'

    ### TRAIN_PARAM ###
    # Image Encoder
    RESIZE = 224
    MODEL_NAME = "tf_efficientnetv2_s_in21ft1k" 
    image_embedding = 1280
    
    pretrained = True
    
    # ProjectionHead
    num_projection_layers = 1
    projection_dim = 512
    dropout = 0.1
    
    # SupConLoss
    temperature = 0.07
    
    # Main
    VISIBLE_GPU = "0"
    MAX_EPOCH = 1
    BATCH_SIZE = 8
    NUM_WORKS = 8
    
    # Re Training
    CON_TRAINING_CHEKPOINT = './ckpt/con_epoch_0.pth'
    CE_TRAINING_CHEKPOINT = './ckpt/ce_epoch_0.pth'
    
    ### INFERNCE_PARAM ###
    BEST_CON_PATH = './ckpt/best_con.pth'
    BEST_CE_PATH = './ckpt/best_ce.pth'
# %%
class Util():
    def encode_label(label, classes_list):
        target = torch.zeros(len(classes_list))
        idx = np.where(classes_list == label)
        target[idx] = 1
        return target

    def acc_func(output, label, batch_size):
        p = torch.argmax(output, dim=1)
        t = torch.argmax(label, dim=1)
        c = (p == t).sum().item()
        acc = c / batch_size
        return acc

    def F2_score(output, label, threshold=0.5, beta=1.0): # if beta > 1 -> Recall에 가중치
        prob = output > threshold
        label = label > threshold

        TP = (prob & label).sum(1).float()
        TN = ((~prob) & (~label)).sum(1).float()
        FP = (prob & (~label)).sum(1).float()
        FN = ((~prob) & label).sum(1).float()

        precision = torch.mean(TP / (TP + FP + 1e-12))
        recall = torch.mean(TP / (TP + FN + 1e-12))
        F2 = (1.0 + (beta**2)) * precision * recall / ((beta**2 * precision) + recall + 1e-12)
        return F2.mean(0)
    
    ### SupCon ###
    def data_transform(df, mode='train'):
        if mode == 'train':
            train_transform = T.Compose([
                T.Resize((CFG.RESIZE, CFG.RESIZE)),
                T.GaussianBlur(kernel_size=(5, 5)), 
                T.RandomVerticalFlip(), 
                T.RandomHorizontalFlip(),
                T.RandomRotation(degrees=(0, 180)),
                T.ToTensor()
            ])
        
            train_dataset = SupConDataset(df, TwoCropTransform(train_transform))
            return train_dataset
        
        elif mode == 'val':
            val_transform = T.Compose([
                T.Resize((CFG.RESIZE, CFG.RESIZE)),
                T.ToTensor()
            ])
            val_dataset = SupConDataset(df, val_transform)
            return val_dataset
        
    def criterion_con(loss_func, outputs, img_labels, batch_size, gpu):
        losses = 0
        for i, key in enumerate(outputs):
            f1, f2 = torch.split(outputs[key], [batch_size, batch_size], dim=0)
            feature = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            losses += loss_func(gpu=gpu,
                                features=feature, 
                                labels=img_labels['labels'][key])
        return losses
    
    def criterion_ce(loss_func, outputs, img_labels, batch_size, gpu):
        losses = 0
        for i, key in enumerate(outputs):
            losses += loss_func(outputs[key], img_labels['labels'][key])
        return losses
        
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        
    def adjust_learning_rate(args, optimizer, epoch):
        lr = args.learning_rate
        if args.cosine:
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / args.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if steps > 0:
                lr = lr * (args.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
        if args.warm and epoch <= args.warm_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / \
                (args.warm_epochs * total_batches)
            lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


    def set_optimizer(opt, model):
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.)
        return optimizer


    def save_model(model, optimizer, epoch, loss, save_file):
        print('==> Saving...')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(state, save_file)
        del state
 
#%%
class TwoCropTransform:
    """Create two transform of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

#%%
class AvgMeter:
    def __init__(self, name="Loss Avg"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
  
#%%
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, gpu, features, labels=None, mask=None):
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # print('features :', features.size())
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda(gpu)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda(gpu)
        else:
            mask = mask.float().cuda(gpu)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print('contrast_feature : ',contrast_feature)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print('anchor_dot_contrast : ',anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(gpu),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        # print('logits :', logits)
        # print('logits :', logits.size())
        # print('logits_mask :', logits_mask)
        # print('logits_mask :', logits_mask.size())
        exp_logits = torch.exp(logits) * logits_mask
        # print('exp_logits :', exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
# %%
class SupConDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = annotations_file[['path', 'level_1', 'level_2', 'level_3']]
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        level_1 = self.img_labels.iloc[idx, 1]
        level_2 = self.img_labels.iloc[idx, 2]
        level_3 = self.img_labels.iloc[idx, 3]
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return {
            'image' : image, 
            'labels' : {
                'level_1' : level_1, 
                'level_2' : level_2, 
                'level_3' : level_3
            }
        }

#%%
class Head(nn.Module):
    def __init__(
        self,
        embedding_dim=CFG.image_embedding,
        projection_dim=CFG.projection_dim,
        target_dim = 10
    ):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gelu = nn.ReLU(inplace=True)
        self.projection = nn.Linear(embedding_dim, projection_dim, bias=True)
        
        self.ce_layer = nn.Linear(projection_dim, target_dim, bias=True) 
    
    def forward(self, x, mode):
        if mode == 'first':
            x = self.fc(x)
            x = self.gelu(x)
            x = self.projection(x)
            # x = self.dropout(x)
            # x = x + projected
            # x = self.layer_norm(x)
            return F.normalize(x)
    
        elif mode == 'second':
            x = self.ce_layer(x)
            return F.normalize(x)
# %%
class Supcon(nn.Module):
    def __init__(self, n_classes_1, n_classes_2, n_classes_3):
        super(Supcon, self).__init__()
        self.model = timm.create_model(model_name=CFG.MODEL_NAME, pretrained=True, num_classes=0, drop_rate=0.3)
        self.model_non_fc = nn.Sequential(*(list(self.model.children())[:-1]))
        
        self.level_1_emb = Head()
        self.level_2_emb = Head()
        self.level_3_emb = Head()
        
        self.level_1_fc = Head(target_dim=n_classes_1)
        self.level_2_fc = Head(target_dim=n_classes_2)
        self.level_3_fc = Head(target_dim=n_classes_3)

    def freeze(self):
        self.model_non_fc.requires_grad_(False)
    
    def encoder(self, x):
        x = self.model_non_fc(x)
        return x
    
    def forward_con(self, x, mode=None):
        x = self.encoder(x)
        return {
            'level_1' : self.level_1_emb(x, mode),
            'level_2' : self.level_2_emb(x, mode),
            'level_3' : self.level_3_emb(x, mode)
        }
            
    def forward_ce(self, x, mode=None):
        x = self.encoder(x)
        return {
            'level_1' : self.level_1_fc(x, mode),
            'level_2' : self.level_2_fc(x, mode),
            'level_3' : self.level_3_fc(x, mode)
        }

# %%
class Training():
    def con_train(dataloader, model, loss_func, optimizer, epoch, max_epoch, gpu, mode):

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
                
                dict_summ = {"loss" : f'{summ["loss"]/(i+1):05.3f}'}
                t.set_postfix(dict_summ)
                t.update()
                
            save_file = './ckpt/con_epoch_{epoch}.pth'.format(epoch=epoch)
            Util.save_model(model, optimizer, epoch, loss, save_file)
                    
        performance_dict.update({key: val/(i+1) for key, val in summ.items()})
        print(performance_dict)
        return model, loss, performance_dict
    
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
                
                output = model(images, mode)

                # Calculate Loss
                batch_size = img_labels['labels']['level_1'].size(0)
                loss = Util.criterion_ce(loss_func, output, img_labels, batch_size, gpu)
                summ["loss"] += loss.item()
                
                # Train & Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
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
                
            save_file = './ckpt/ce_epoch_{epoch}.pth'.format(epoch=epoch)
            Util.save_model(model, optimizer, epoch, loss, save_file)
        
        performance_dict.update({key: val/(i+1) for key, val in summ.items()})
        print(performance_dict)
        return loss, performance_dict

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
                    loss = Util.criterion_ce(loss_func, output, img_labels)
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
        return loss, performance_dict

# %%
class Start():
    def __init__(self):
        self.best_val_loss = np.inf
        self.best_val_acc = 0

    def StartTraining(self, release_mode=False):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        cudnn.deterministic = True
        # To do : test cudnn.benchmark = False for reproductible result despite speed down
        cudnn.benchmark = True
        warnings.warn('''You have chosen to seed training. 
                        This will turn on the CUDNN deterministic setting, 
                        which can slow down your training considerably! 
                        You may see unexpected behavior when restarting 
                        from checkpoints.''')

        if CFG.VISIBLE_GPU is not None:
            warnings.warn('''You have chosen a specific GPU. This will completely 
                            disable data parallelism.''')

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=CFG.VISIBLE_GPU
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node > 1:
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            # torch.cuda.manual_seed_all(42) # if use multi-GPU
            mp.spawn(self.main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, release_mode))
        else:
            # Simply call main_worker function
            self.main_worker(int(CFG.VISIBLE_GPU), ngpus_per_node, release_mode)

    def main_worker(self, gpu, ngpus_per_node, release_mode):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        warnings.filterwarnings('ignore')

        if gpu is not None:
            print("Use GPU: {} for training".format(gpu))

        if ngpus_per_node > 1:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:2222",
                                    world_size=ngpus_per_node, rank=gpu)
        
        # DataSet Load
        train_df = pd.read_csv('D:/my_git/doing/train_df.csv')
        val_df = pd.read_csv('D:/my_git/doing/val_df.csv')

        # Train Data Load
        train_dataset = Util.data_transform(train_df , mode='train')

        if ngpus_per_node > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=(train_sampler is None), 
            num_workers=CFG.NUM_WORKS, pin_memory=True, sampler=train_sampler
        )

        # Validation Data Load
        val_dataset = Util.data_transform(val_df, mode='val')
        val_loader = DataLoader(
            val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, # False 
            num_workers=CFG.NUM_WORKS, pin_memory=True
        )

        if (ngpus_per_node == 1) or (ngpus_per_node > 1
                and gpu % ngpus_per_node == 0):
            print("PyTorch Version :", torch.__version__)
            
        n_classes_1 = len(train_df['level_1'].unique())
        n_classes_2 = len(train_df['level_2'].unique())
        n_classes_3 = len(train_df['level_3'].unique())
        print('Num classes_1 :', n_classes_1)
        print('Num classes_2 :', n_classes_2)
        print('Num classes_3 :', n_classes_3)

        # Create Model
        model = Supcon(n_classes_1, n_classes_2, n_classes_3)
        
        # define loss function (criterion) and optimizer
        loss_con = SupConLoss().cuda(gpu)
        loss_ce = nn.CrossEntropyLoss().cuda(gpu)
    
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Scheduler
        def adjust_learning_rate(optimizer, epoch, ngpus_per_node):
            if epoch < 2:
                lr = .001*ngpus_per_node
            elif epoch < 4:
                lr = .0007*ngpus_per_node
            elif epoch < 6:
                lr = .0004*ngpus_per_node
            elif epoch < 8:
                lr = .0002*ngpus_per_node
            elif epoch < 10:
                lr = .0001*ngpus_per_node
            elif epoch < 50:
                lr = .00005*ngpus_per_node
            elif epoch < 55:
                lr = .00001*ngpus_per_node
            else:
                lr = .000001*ngpus_per_node
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # DistributedDataParallel
        model = self.ddp(model, ngpus_per_node, gpu)
        
        # Inference
        # if release_mode == 'inference':
        #     checkpoint = torch.load(CFG.BEST_CLIP_PATH)
        #     model.load_state_dict(checkpoint)
            
        #     metrics = Inference(model=model, 
        #                         val_loader=val_loader, 
        #                         gpu=gpu).inference()
            
        #     with open('CLIP_metrics_inference.pickle', 'wb') as fw:
        #         pickle.dump(metrics, fw)
        #     return ###
        
        # # Re-Training 
        if release_mode == 'retrain':
            checkpoint = torch.load(CFG.TRAINING_CHEKPOINT)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        
        # Loss가 Nan값이 되는 부분 탐지
        # torch.autograd.set_detect_anomaly(True)
        
        # Training Start -!
        metrics_train = []
        metrics_val = []
        for epoch in range(CFG.MAX_EPOCH):
            # for shuffling
            if ngpus_per_node > 1:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, ngpus_per_node)
            
            lr = optimizer.param_groups[0]["lr"] / ngpus_per_node

            # train for one epoch
            mode = 'first'
            model, con_loss, train_metrics_summary = Training.con_train(train_loader, model, loss_con, optimizer, epoch, CFG.MAX_EPOCH, gpu, mode)
            metrics_train.append(train_metrics_summary)

            # Best Model Save
            if con_loss.avg < self.best_val_loss:
                self.best_val_loss = con_loss.avg
                torch.save(model.state_dict(), CFG.BEST_CON_PATH)
                print('Save Best Model -!', epoch)
            
        #########################################################
        
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        checkpoint = torch.load(CFG.BEST_CON_PATH)
        model.load_state_dict(checkpoint)
        model.freeze()
        
        metrics_train = []
        metrics_val = []
        for epoch in range(CFG.MAX_EPOCH):
            # for shuffling
            if ngpus_per_node > 1:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, ngpus_per_node)
            
            lr = optimizer.param_groups[0]["lr"] / ngpus_per_node

            # train for one epoch
            mode = 'second'
            train_metrics_summary = Training.ce_train(train_loader, model, loss_ce, optimizer, epoch, CFG.MAX_EPOCH, gpu, mode)
            metrics_train.append(train_metrics_summary)
            
            # evaluate on validation set
            val_loss, val_metrics_summary = Training.eval(val_loader, model, loss_ce, epoch, CFG.MAX_EPOCH, gpu)
            metrics_val.append(val_metrics_summary)
            
            # Best Model Save
            if val_loss.avg < self.best_val_loss:
                self.best_val_loss = val_loss.avg
                torch.save(model.state_dict(), CFG.BEST_CE_PATH)
                print('Save Best Model -!', epoch)
        
    def ddp(self, model, ngpus_per_node, gpu):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        elif ngpus_per_node > 1:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if gpu is not None:
                torch.cuda.set_device(gpu)
                model.cuda(gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True) # find_unused_parameters=True
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True) # find_unused_parameters=True
        elif gpu is not None:
            gpu = 0
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if CFG.MODEL_NAME.startswith('alexnet') or CFG.MODEL_NAME.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        print('Number of Parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model
                
# %%
if __name__ == "__main__":
    start = Start()
    start.StartTraining(release_mode=CFG.release_mode)
