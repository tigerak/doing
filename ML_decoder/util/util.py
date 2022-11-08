import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset

class Util():
    def encode_label(label, classes_list):
        target = torch.zeros(len(classes_list))
        idx = np.where(classes_list == label)
        target[idx] = 1
        return target

    # def decode_target(target, classes_list, threshold=0.5):
    #     result = []
    #     for i, x in enumerate(target):
    #         if (x >= threshold):
    #             result.append(classes_list[i])
    #     return result

    def criterion(loss_func, outputs, img_labels):
        losses = 0
        for i, key in enumerate(outputs):
            losses += loss_func(outputs[key], img_labels['labels'][key])
        return losses

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
    
class ImageDataset(Dataset):
    def __init__(self, annotations_file, class_list_1, class_list_2, class_list_3, transform=None):
        self.img_labels = annotations_file[['path', 'idx_1', 'idx_2', 'idx_3']]
        self.class_list_1 = class_list_1
        self.class_list_2 = class_list_2
        self.class_list_3 = class_list_3
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label_1 = self.img_labels.iloc[idx, 1]
        label_2 = self.img_labels.iloc[idx, 2]
        label_3 = self.img_labels.iloc[idx, 3]
        
        return {
            'image' : image, 
            'labels' : {
                'level_1' : Util.encode_label(label_1, self.class_list_1), 
                'level_2' : Util.encode_label(label_2, self.class_list_2), 
                'level_3' : Util.encode_label(label_3, self.class_list_3)
            }
        }
