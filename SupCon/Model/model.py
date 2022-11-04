import torch.nn.functional as F
import torch.nn as nn

import timm

from cfg import CFG

class Head(nn.Module):
    def __init__(
        self,
        embedding_dim=CFG.image_embedding,
        projection_dim=CFG.projection_dim,
        target_dim = 10
    ):
        super().__init__()
        # first
        self.fc = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gelu = nn.ReLU(inplace=True)
        self.projection = nn.Linear(embedding_dim, projection_dim, bias=True)
        # second
        self.ce_layer = nn.Linear(embedding_dim, target_dim, bias=True) 
    
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