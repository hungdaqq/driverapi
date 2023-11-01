import sys
import torch
import torch.nn as nn

sys.path.append('models')
from models.mobilevitv1 import MobileViT
from models.mobilevitv2 import MobileViTv2

image_size = (256,256)
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileViT(
    image_size=image_size,
    mode='small',
    num_classes=1000,
    patch_size=(2,2)
)
print(model)
pretrained = 'pretrained/mobilevit_s.pt'
state_dict = torch.load(pretrained, map_location=device)
model.load_state_dict(state_dict)

model.classifier.fc = nn.Linear(in_features=model.classifier.fc.in_features, out_features=num_classes, bias=True)

# model.load_state_dict(state_dict)

# model = model.to(device)