import numpy as np
import math
import os

from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from cfg import CFG


class TwoCropTransform:
    """Create two transform of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class SupConDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = annotations_file[['path', 'idx_1', 'idx_2', 'idx_3']]
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
        
        
class Util():
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
    
    def criterion_ce(loss_func, outputs, img_labels):
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
        if not os.path.exists(CFG.ROOT_CHEKPOINT):
            os.mkdir(CFG.ROOT_CHEKPOINT)
        print('==> Saving...')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(state, save_file)
        del state
        
        
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