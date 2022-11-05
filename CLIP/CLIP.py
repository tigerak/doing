# %%
import enum
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

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import itertools


# %%
class CFG:
    release_mode = 'retrain' # None, 'inference', 'retrain'
    DB_PATH = '/home/doinglab-hs/ak/db/eng_df.csv'

    ### TRAIN_PARAM ###
    # Image Encoder
    RESIZE = 300
    MODEL_NAME = "tf_efficientnetv2_s_in21ft1k" 
    image_embedding = 1280
    image_encoder_lr = 1e-4
    
    # Text Encoder
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200
    text_encoder_lr = 1e-5
    
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    
    # ProjectionHead
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    
    # CLIP
    temperature = 10.0
    
    # Optimizer Param
    head_lr = 1e-3
    weight_decay = 1e-3
    
    # lr_scheduler
    patience = 1
    factor = 0.8
    
    # Main
    VISIBLE_GPU = "0, 1"
    MAX_EPOCH = 20
    BATCH_SIZE = 8
    NUM_WORKS = 8
    
    # Re Training
    TRAINING_CHEKPOINT = '/home/doinglab-hs/ak/CLIP_train/training_model.pt'
    
    ### INFERNCE_PARAM ###
    BEST_CLIP_PATH = '/home/doinglab-hs/ak/best_CLIP/best_CLIP.pt'
    CANDIDATE_NUM = 3
    TOP_K_ACC = 3
# %%
class Util():
    def encode_label(label, classes_list):
        target = torch.zeros(len(classes_list))
        idx = np.where(classes_list == label)
        target[idx] = 1
        return target

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
    
    def topk_acc(pred, target, top_k_num):
        if top_k_num == 1:
            if target == pred[0]:
                return 1
            else:
                return 0
        elif top_k_num > 1:
            if target in pred:
                return 1
            else:
                return 0

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

    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    
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
# %%
class CLIPDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transform=None):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {
            key : torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        item['image'] = image.clone().detach().float()
        item['caption'] = self.captions[idx]
        
        return item

# %%
class ImageEncoder(nn.Module):
    def __init__(self, 
                 model_name=CFG.MODEL_NAME, 
                 pretrained=CFG.pretrained, 
                 trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, 
                                       pretrained=pretrained, 
                                       num_classes=0, 
                                       drop_rate=0.3)
        for p in self.model.parameters():
            p.requires_grad = trainable
            
    def forward(self, x):
        return torch.nan_to_num(self.model(x)) # RuntimeError: Function 'MulBackward0' returned nan values in its 1th output.

#%%
class TextEncoder(nn.Module):
    def __init__(self, 
                 model_name=CFG.text_encoder_model, 
                 pretrained=CFG.pretrained, 
                 trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return torch.nan_to_num(last_hidden_state[:, self.target_token_idx, :])
    
#%%
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(CFG.dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

#%%
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = Util.cross_entropy(logits, targets, reduction='none')
        images_loss = Util.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        return loss.mean(), logits, targets

# %%
class Training():
    def train(dataloader, model, optimizer, lr_scheduler, step, epoch, max_epoch, gpu):

        model.train()

        performance_dict = {
            "epoch": epoch+1
        }

        summ = {
            "loss": 0,
            "F2" : 0,
            "Acc" : 0
        }

        loss_meter = AvgMeter()
        
        with tqdm(total=len(dataloader)) as t:
            t.set_description(f'[{epoch+1}/{max_epoch}]')
            
            # Iteration step
            for i, batch in enumerate(dataloader):
                
                if gpu is not None:
                    batch = {k: v.cuda(gpu, non_blocking=True) for k, v in batch.items() if k != "caption"}
                    
                loss, logits, targets = model(batch)
                
                summ["loss"] += loss.item()
                
                # Train & Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step == "batch":
                    lr_scheduler.step()
                    
                count = batch["image"].size(0)
                loss_meter.update(loss.item(), count)

                summ["F2"] += Util.F2_score(logits, targets)
                summ["Acc"] += Util.acc_func(logits, targets, count)

                dict_summ = {"loss" : f'{summ["loss"]/(i+1):05.3f}'}
                dict_summ.update({"F2" : f'{summ["F2"]/(i+1):05.3f}'})
                dict_summ.update({"Acc" : f'{summ["Acc"]/(i+1)*100:05.3f}'})
                t.set_postfix(dict_summ)
                t.update()
        
        # Save model each Epoch -!
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(), #.module
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            }, CFG.TRAINING_CHEKPOINT)
        print('training_model Save -!')
        
        performance_dict.update({key: val/(i+1) for key, val in summ.items()})
        print(performance_dict)
        return loss_meter, performance_dict

    def eval(dataloader, model, epoch, max_epoch, gpu):

        model.eval()
        
        performance_dict = {
            "epoch": epoch+1
        }

        summ = {
            "loss_val": 0,
            "F2_val" : 0,
            "Acc_val": 0
        }

        loss_meter = AvgMeter()
        
        with tqdm(total=len(dataloader)) as t:
            with torch.no_grad():
                t.set_description(f'[{epoch+1}/{max_epoch}]')
                # Iteration step
                for i, batch in enumerate(dataloader):

                    if gpu is not None:
                        batch = {k: v.cuda(gpu, non_blocking=True) for k, v in batch.items() if k != "caption"}

                    loss, logits, targets = model(batch)
                    
                    # Calculate Loss
                    summ["loss_val"] += loss.item()

                    # Calculate Metrics
                    count = batch['image'].size(0)
                    loss_meter.update(loss.item(), count)
                    
                    summ["F2_val"] += Util.F2_score(logits, targets)
                    summ["Acc_val"] += Util.acc_func(logits, targets, count)

                    dict_summ = {"loss_val" : f'{summ["loss_val"]/(i+1):05.3f}'}
                    dict_summ.update({"F2_val" : f'{summ["F2_val"]/(i+1):05.3f}'})
                    dict_summ.update({"Acc_val" : f'{summ["Acc_val"]/(i+1)*100:05.3f}'})
                    t.set_postfix(dict_summ)
                    t.update()
        
        performance_dict.update({key : val/(i+1) for key, val in summ.items()})
        print(performance_dict)
        return loss_meter, performance_dict

#%%
class Inference():
    def __init__(self, model, val_loader, gpu):
        super().__init__()
        self.model = model
        self.val_loader = val_loader
        self.gpu = gpu
        
        self.best_val_loss = np.inf
        self.best_val_acc = 0
        
    def get_all_text_embeddings(self):
        
        self.model.eval()
        
        all_text_embeddings = {}
        with torch.no_grad():
            for batch_raw in tqdm(self.val_loader):
                if self.gpu is not None:
                    batch = {k: v.cuda(self.gpu, non_blocking=True) for k, v in batch_raw.items() if k != "caption"}
                text_features = self.model.module.text_encoder(batch['input_ids'], 
                                                               batch['attention_mask'])
                text_embeddings = self.model.module.text_projection(text_features)
                for i, cap in enumerate(batch_raw['caption']):
                    if cap not in all_text_embeddings.keys():
                        all_text_embeddings[cap] = text_embeddings[i]
                
        with open('all_text_embeddings.pickle', 'wb') as fw:
            pickle.dump(all_text_embeddings, fw)
            
        return all_text_embeddings
    
    def get_image_embedding(self, image):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        warnings.filterwarnings('ignore')
        
        # get Image Embedding
        image_feature = self.model.module.image_encoder(image)
        image_embedding = self.model.module.image_projection(image_feature)
        
        return image_embedding
    
    def inference(self, n=CFG.CANDIDATE_NUM):
        # make all text embedding dictiona
        if not os.path.exists('/home/doinglab-hs/ak/all_text_embeddings.pickle'):
            all_text_embeddings = self.get_all_text_embeddings()
        else:
            with open('/home/doinglab-hs/ak/all_text_embeddings.pickle', 'rb') as f:
                all_text_embeddings = pickle.load(f) 
        
        emb_list = []
        cap_list = []
        for i, (k, v) in enumerate(all_text_embeddings.items()):
            cap_list.append(k)
            emb_list.append(v.cpu().numpy())
        emb_list = torch.Tensor(emb_list).cuda(self.gpu, non_blocking=True)
        
        metrics = {
            'Top_1_Acc' : 0,
            f'Top_{CFG.TOP_K_ACC}_Acc' : 0
        }
        
        with tqdm(total=len(self.val_loader)) as t:
            with torch.no_grad():
                for i, batch_raw in enumerate(self.val_loader):
                    if self.gpu is not None:
                        batch = {k: v.cuda(self.gpu, non_blocking=True) for k, v in batch_raw.items() if k != "caption"}
                    image_embedding = self.get_image_embedding(batch['image'])

                    image_embeddings_n = F.normalize(image_embedding, p=2, dim=-1)
                    text_embeddings_n = F.normalize(emb_list, p=2, dim=-1)
                    dot_similarity = image_embeddings_n @ text_embeddings_n.T
                    
                    values, indices = torch.topk(dot_similarity.squeeze(0), n)
                    print(indices)
                    
                    top_1_summ = 0
                    top_k_summ = 0
                    for j, b in enumerate(indices):
                        pred = [cap_list[idx] for idx in b]
                        target = batch_raw['caption'][j]
                        print(pred)
                        print(target)
                        top_1_summ += Util.topk_acc(pred, target, 1)
                        top_k_summ += Util.topk_acc(pred, target, CFG.TOP_K_ACC)
                        
                    metrics["Top_1_Acc"] += top_1_summ / indices.size()[0]
                    metrics[f"Top_{CFG.TOP_K_ACC}_Acc"] += top_k_summ / indices.size()[0]
                    
                    dict_metrics = {'Top_1_Acc' : f'{metrics["Top_1_Acc"]/(i+1)*100:05.3f}'}
                    dict_metrics.update({f"Top_{CFG.TOP_K_ACC}_Acc" : f'{metrics[f"Top_{CFG.TOP_K_ACC}_Acc"]/(i+1)*100:05.3f}'})
                    t.set_postfix(dict_metrics)
                    t.update()
            
        return metrics

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
        annotations_df = pd.read_csv(CFG.DB_PATH)
        train_df = annotations_df[annotations_df['type']=='TRAIN']
        val_df = annotations_df[annotations_df['type']=='VAL']

        # Train Data Load
        train_transform = T.Compose([
            T.Resize((CFG.RESIZE, CFG.RESIZE)),
            T.RandomAffine(degrees=90, translate=(.12,.12), scale=(.85, 1.15), shear=.18, fill=255),
            T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = CLIPDataset(
            train_df['file_name'].values, 
            train_df['caption'].values, 
            tokenizer=DistilBertTokenizer.from_pretrained(CFG.text_tokenizer),
            transform=train_transform)

        if ngpus_per_node > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=(train_sampler is None), 
            num_workers=CFG.NUM_WORKS, pin_memory=True, sampler=train_sampler
        )

        # Validation Data Load
        val_transform = T.Compose([
            T.Resize((CFG.RESIZE, CFG.RESIZE)),
            T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_dataset = CLIPDataset(
            val_df['file_name'].values, 
            val_df['caption'].values, 
            tokenizer=DistilBertTokenizer.from_pretrained(CFG.text_tokenizer),
            transform=val_transform
        )
        val_loader = DataLoader(
            val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, # False  
            num_workers=CFG.NUM_WORKS, pin_memory=True
        )

        if (ngpus_per_node == 1) or (ngpus_per_node > 1
                and gpu % ngpus_per_node == 0):
            print("PyTorch Version :", torch.__version__)

        # Create Model
        model = CLIPModel()
        
        # Optimizer
        params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=CFG.weight_decay)
        
        # Scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
        )
        step = "epoch"
        
        # DistributedDataParallel
        model = self.ddp(model, ngpus_per_node, gpu)
        
        # Inference
        if release_mode == 'inference':
            checkpoint = torch.load(CFG.BEST_CLIP_PATH)
            model.load_state_dict(checkpoint)
            
            metrics = Inference(model=model, 
                                val_loader=val_loader, 
                                gpu=gpu).inference()
            
            with open('CLIP_metrics_inference.pickle', 'wb') as fw:
                pickle.dump(metrics, fw)
            return ###
        
        # # Re-Training 
        if release_mode == 'retrain':
            checkpoint = torch.load(CFG.TRAINING_CHEKPOINT)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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

            # train for one epoch
            train_loss, train_metrics_summary = Training.train(train_loader, model, optimizer, lr_scheduler, step, epoch, CFG.MAX_EPOCH, gpu)
            metrics_train.append(train_metrics_summary)

            # evaluate on validation set
            val_loss, val_metrics_summary = Training.eval(val_loader, model, epoch, CFG.MAX_EPOCH, gpu)
            metrics_val.append(val_metrics_summary)
            
            # Best Model Save
            if val_loss.avg < self.best_val_loss:
                self.best_val_loss = val_loss.avg
                torch.save(model.state_dict(), CFG.BEST_CLIP_PATH)
                print('Save Best Model -!', epoch)

        lr_scheduler.step(val_loss.avg)
        
        with open('CLIP_metrics_train.pickle', 'wb') as fw:
            pickle.dump(metrics_train, fw)
        with open('CLIP_metrics_val.pickle', 'wb') as fw:
            pickle.dump(metrics_val, fw)
           
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
# %%
