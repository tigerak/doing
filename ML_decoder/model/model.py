#%%
import torch.nn as nn

import timm

from cfg import CFG

print(CFG.MODEL_NAME)
class MultilabelImageClassification(nn.Module):
    def __init__(self, n_level_1, n_level_2, n_level_3):
        super(MultilabelImageClassification, self).__init__()
        self.model = timm.create_model(model_name="tf_efficientnetv2_s_in21ft1k", pretrained=True, num_classes=0, drop_rate=0.3)
        self.model_non_fc = nn.Sequential(*(list(self.model.children())[:-2]))
        
        self.head = nn.Sequential(
            
        )

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
        
#%%
import timm

from ml_decoder import add_ml_decoder_head

model = timm.create_model(model_name="tf_efficientnetv2_s_in21ft1k", pretrained=True, num_classes=0, drop_rate=0.3)

model = add_ml_decoder_head(model=model,
                            num_classes=2424,
                            num_of_groups=3)

print(model)