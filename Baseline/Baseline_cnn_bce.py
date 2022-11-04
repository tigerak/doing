# %%
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
    #TRAIN_PATH
    ROOT = "/disk1/first/Food/tree_220915/"
    TRAIN = "train/"
    VAL = "validation/"
    
    #"TRAIN_PARAM
    RESIZE = 300
    MODEL_NAME = "tf_efficientnetv2_m_in21ft1k" 
    VISIBLE_GPU = "0, 1"
    MAX_EPOCH = 20
    BATCH_SIZE = 8
    NUM_WORKS = 8

# %%
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

# %%
class ImageDataset(Dataset):
    def __init__(self, annotations_file, class_list_1, class_list_2, class_list_3, transform=None):
        self.img_labels = annotations_file[['file_name', 'level_1', 'level_2', 'level_3']]
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

# %%
class MultilabelImageClassification(nn.Module):
    def __init__(self, n_level_1, n_level_2, n_level_3):
        super(MultilabelImageClassification, self).__init__()
        self.model = timm.create_model(model_name=CFG.MODEL_NAME, pretrained=True, num_classes=0, drop_rate=0.3)
        self.model_non_fc = nn.Sequential(*(list(self.model.children())[:-1]))

        self.level_1_fc = nn.Sequential(
            nn.Linear(in_features=1280, out_features=n_level_1, bias=True)
        )
        self.level_2_fc = nn.Sequential(
            nn.Linear(in_features=1280, out_features=n_level_2, bias=True)
        )
        self.level_3_fc = nn.Sequential(
            nn.Linear(in_features=1280, out_features=n_level_3, bias=True)
        )

    def forward(self, x):
        x = self.model_non_fc(x)

        return {
            'level_1' : self.level_1_fc(x),
            'level_2' : self.level_2_fc(x),
            'level_3' : self.level_3_fc(x)
        }

# %%
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
# %%
best_val_loss = np.inf
best_val_acc = 0

def StartTraining(release_mode=False):
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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, release_mode))
    else:
        # Simply call main_worker function
        main_worker(int(CFG.VISIBLE_GPU), ngpus_per_node, release_mode)

def main_worker(gpu, ngpus_per_node, release_mode):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    warnings.filterwarnings('ignore')
    global best_val_loss
    global best_val_acc

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if ngpus_per_node > 1:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:2222",
                                world_size=ngpus_per_node, rank=gpu)
    
    # DataSet & DataLoader
    annotations_df = pd.read_csv('/home/doinglab-hs/ak/annotations_df.csv')
    train_df = annotations_df[annotations_df['type']=='TRAIN']
    val_df = annotations_df[annotations_df['type']=='VAL']

    class_list_1 = train_df['level_1'].unique()
    class_list_2 = train_df['level_2'].unique()
    class_list_3 = train_df['level_3'].unique()
    n_classes_1 = len(class_list_1)
    n_classes_2 = len(class_list_2)
    n_classes_3 = len(class_list_3)
    print(n_classes_1, n_classes_2, n_classes_3)

    # Train Data Loader
    train_transform = T.Compose([
        # T.ToPILImage(mode='RGB'),
        T.Resize((CFG.RESIZE, CFG.RESIZE)),
        T.RandomAffine(degrees=90, translate=(.12,.12), scale=(.85, 1.15), shear=.18, fill=255),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = ImageDataset(train_df, class_list_1, class_list_2, class_list_3, transform=train_transform)

    if ngpus_per_node > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=(train_sampler is None), 
        num_workers=CFG.NUM_WORKS, pin_memory=True, sampler=train_sampler
    )

    # Validation Data Loader
    val_transform = T.Compose([
        T.Resize((CFG.RESIZE, CFG.RESIZE)),
        T.ToTensor()
    ])
    val_dataset = ImageDataset(val_df, class_list_1, class_list_2, class_list_3, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, 
        num_workers=CFG.NUM_WORKS, pin_memory=True
    )

    if (ngpus_per_node == 1) or (ngpus_per_node > 1
            and gpu % ngpus_per_node == 0):
        print("PyTorch Version :", torch.__version__)

    # Create Model
    model = MultilabelImageClassification(n_classes_1, n_classes_2, n_classes_3)
    
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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False) # find_unused_parameters=True
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False) # find_unused_parameters=True
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

    # Loss function
    loss_func = nn.BCEWithLogitsLoss().cuda(gpu)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

     # scheduler
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
    
    metrics_train = []
    metrics_val = []
    for epoch in range(CFG.MAX_EPOCH):
        # for shuffling
        if ngpus_per_node > 1:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, ngpus_per_node)

        lr = optimizer.param_groups[0]["lr"] / ngpus_per_node

        # train for one epoch
        metrics_summary = Training.train(train_loader, model, loss_func, optimizer, epoch, CFG.MAX_EPOCH, gpu)
        metrics_train.append(metrics_summary)

        # evaluate on validation set
        metrics_summary.update(Training.eval(val_loader, model, loss_func, epoch, CFG.MAX_EPOCH, gpu))
        metrics_val.append(metrics_summary)

        # is_best_loss = metrics_summary["loss_val"] < best_val_loss
        # is_best_acc = metrics_summary["acc_val"] > best_val_acc

    with open('bcewl_metrics_train.pickle', 'wb') as fw:
        pickle.dump(metrics_train, fw)
    with open('bcewl_metrics_val.pickle', 'wb') as fw:
        pickle.dump(metrics_val, fw)
    
# %%
if __name__ == "__main__":
    StartTraining()
# %%
