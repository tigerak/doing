import torch.nn as nn
import timm

from model.ml_decoder import MLDecoder
from cfg import CFG

class MLDcoderClassification(nn.Module):
    def __init__(self, n_level_1, n_level_2, n_level_3):
        super(MLDcoderClassification, self).__init__()
        self.backborn = timm.create_model(model_name=CFG.MODEL_NAME, pretrained=True, num_classes=0, drop_rate=0.3)
        self.backborn_non_gap = nn.Sequential(*(list(self.backborn.children())[:-2]))
        
        self.level_1_decoder = nn.Sequential(
            nn.Identity(),
            MLDecoder(num_classes=n_level_1,
                      initial_num_features=1280,
                      num_of_groups=1,
                      decoder_embedding=768,
                      zsl=0)
        )
        self.level_2_decoder = nn.Sequential(
            nn.Identity(),
            MLDecoder(num_classes=n_level_2,
                      initial_num_features=1280,
                      num_of_groups=1,
                      decoder_embedding=768,
                      zsl=0)
        )
        self.level_3_decoder = nn.Sequential(
            nn.Identity(),
            MLDecoder(num_classes=n_level_3,
                      initial_num_features=1280,
                      num_of_groups=1,
                      decoder_embedding=768,
                      zsl=0)
        )
        
    def forward(self, x):
        x = self.backborn_non_gap(x)

        return {
            'level_1' : self.level_1_decoder(x),
            'level_2' : self.level_2_decoder(x),
            'level_3' : self.level_3_decoder(x)
        }