import numpy as np
import pandas as pd
import os
import pickle

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
from torch.utils.data import DataLoader
import torchvision.transforms as T

from cfg import CFG
from util.util import ImageDataset
from model.model import MultilabelImageClassification
from util.train import Training

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
    train_df = pd.read_csv('D:/my_git/doing/train_df.csv')
    val_df = pd.read_csv('D:/my_git/doing/val_df.csv')

    class_list_1 = train_df['idx_1'].unique()
    class_list_2 = train_df['idx_2'].unique()
    class_list_3 = train_df['idx_3'].unique()
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