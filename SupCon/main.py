import numpy as np
import pandas as pd
import os

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

from cfg import CFG
from Util.util import Util 
from Util.loss import SupConLoss
from Util.train import Training
from Model.model import Supcon


class Start():
    def __init__(self):
        self.best_con_loss = np.inf
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
            
        n_classes_1 = len(train_df['idx_1'].unique())
        n_classes_2 = len(train_df['idx_2'].unique())
        n_classes_3 = len(train_df['idx_3'].unique())
        print('Num classes_1 :', n_classes_1)
        print('Num classes_2 :', n_classes_2)
        print('Num classes_3 :', n_classes_3)

        # Create Model
        model = Supcon(n_classes_1, n_classes_2, n_classes_3)
        
        # define loss function (criterion) and optimizer
        loss_con = SupConLoss().cuda(gpu)
        loss_ce = nn.CrossEntropyLoss().cuda(gpu)
    
        # Optimizer
        optimizer_con = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_ce = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # DistributedDataParallel
        model = self.ddp(model, ngpus_per_node, gpu)
        
        # Inference
        # if release_mode == 'inference':
        #     checkpoint = torch.load(CFG.BEST_CE_PATH)
        #     model.load_state_dict(checkpoint)
            
        #     metrics = Inference(model=model, 
        #                         val_loader=val_loader, 
        #                         gpu=gpu).inference()
            
        #     with open('CLIP_metrics_inference.pickle', 'wb') as fw:
        #         pickle.dump(metrics, fw)
        #     return ###
        
        if 'first' in CFG.LEARNING_STEP:
            max_epoch = CFG.MAX_EPOCH 
            # # Re-Training 
            if release_mode == 'retrain_con':
                checkpoint = torch.load(CFG.CON_TRAINING_CHEKPOINT)
                model.load_state_dict(checkpoint['model'])
                optimizer_con.load_state_dict(checkpoint['optimizer'])
                max_epoch = CFG.MAX_EPOCH - checkpoint['epoch']
                loss = checkpoint['loss']
            
            # Loss가 Nan값이 되는 부분 탐지
            # torch.autograd.set_detect_anomaly(True)
            
            # First Training -!
            metrics_train = []
            metrics_val = []
            for epoch in range(max_epoch):
                if release_mode == 'retrain_con':
                    epoch = epoch + checkpoint['epoch']
                # for shuffling
                if ngpus_per_node > 1:
                    train_sampler.set_epoch(epoch)
                self.adjust_learning_rate(optimizer_con, epoch, ngpus_per_node)
                
                lr = optimizer_con.param_groups[0]["lr"] / ngpus_per_node

                # train for one epoch
                mode = 'first'
                con_loss, train_metrics_summary = Training.con_train(train_loader, model, loss_con, optimizer_con, epoch, max_epoch, gpu, mode)
                metrics_train.append(train_metrics_summary)

                # Best Model Save
                if con_loss.avg < self.best_con_loss:
                    self.best_con_loss = con_loss.avg
                    torch.save(model.state_dict(), CFG.BEST_CON_PATH)
                    print('Save Best Model -!', epoch)
            
        if 'second' in CFG.LEARNING_STEP:
            # Load checkpoint.
            print("==> Resuming from checkpoint..")
            checkpoint = torch.load(CFG.BEST_CON_PATH)
            model.load_state_dict(checkpoint)
            model.freeze()
            
            # Second Training -!
            metrics_train = []
            metrics_val = []
            for epoch in range(CFG.MAX_EPOCH):
                # for shuffling
                if ngpus_per_node > 1:
                    train_sampler.set_epoch(epoch)
                self.adjust_learning_rate(optimizer_ce, epoch, ngpus_per_node)
                
                lr = optimizer_ce.param_groups[0]["lr"] / ngpus_per_node

                # train for one epoch
                mode = 'second'
                ce_loss, train_metrics_summary = Training.ce_train(train_loader, model, loss_ce, optimizer_ce, epoch, CFG.MAX_EPOCH, gpu, mode)
                metrics_train.append(train_metrics_summary)
                
                # evaluate on validation set
                val_loss, val_metrics_summary = Training.eval(val_loader, model, loss_ce, epoch, CFG.MAX_EPOCH, gpu, mode)
                metrics_val.append(val_metrics_summary)
                
                # Best Model Save
                if val_loss.avg < self.best_val_loss:
                    self.best_val_loss = val_loss.avg
                    torch.save(model.state_dict(), CFG.BEST_CE_PATH)
                    print('Save Best Model -!', epoch)
        
    # Scheduler
    def adjust_learning_rate(self, optimizer, epoch, ngpus_per_node):
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
