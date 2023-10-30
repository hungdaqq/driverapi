import torch
import sys

sys.path.append('models')
from models.mobilevitv1 import MobileViT
from models.mobilevitv2 import MobileViTv2

# Initialize the model
model = MobileViT(
    image_size=(256,256),
    mode='xx_small',
    num_classes=1000,
    patch_size=(2,2)
)
# model = MobileViTv2(
#     image_size=(256,256),
#     width_multiplier=0.5,
#     num_classes=1000,
#     patch_size=(2,2)
# )

print(model)
# Load the saved weights
device = torch.device('cuda')

state_dict = torch.load('pretrained/mobilevit_xxs.pt', map_location=device)
# print(state_dict)
# print(state_dict.keys())
# Load the state dict into the model
model.load_state_dict(state_dict)
# print(model.state_dict())
