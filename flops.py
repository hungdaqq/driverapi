import sys
import torch
import torch.nn as nn
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append('models')
from models.mobilevitv1 import MobileViT
from models.mobilevitv2 import MobileViTv2

from fvcore.nn import FlopCountAnalysis, flop_count_table 

image_size = (256,256)
num_classes = 10

# Initialize the model

# model = MobileViT(
#     image_size=image_size,
#     mode='xx_small',
#     num_classes=num_classes,
#     patch_size=(2,2)
# )

# model = MobileViTv2(
#     image_size=image_size,
#     width_multiplier=0.5,
#     num_classes=num_classes,
#     patch_size=(2,2)
# )

# Define your model architecture (make sure it matches the one used during training)

# # EffNetB0
# model = models.efficientnet_b0()
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=False),
#     nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
# )

# MNASNET0_5
# model = models.mnasnet0_5()
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=False),
#     nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
# )

# # MOBILENET_V2
# model = models.mobilenet_v2()
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=False),
#     nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
# )

# MOBILENET_V3_SMALL
model = models.mobilenet_v3_small()
model.classifier = nn.Sequential(
    nn.Linear(in_features=model.classifier[0].in_features, out_features=model.classifier[0].out_features, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes, bias=True)
)

x = torch.randn(5, 3, 256, 256)

flop_analyzer = FlopCountAnalysis(model, x)
print(flop_count_table(flop_analyzer))