import torch
from mobilevitv1 import MobileViT
import collections
# Initialize the model
model = MobileViT(
    image_size=(256,256),
    mode='xx_small',
    num_classes=10,
    patch_size=(2,2)
)

# Load the saved weights
device = torch.device('cuda')

state_dict = torch.load('mobilevit_xxs.pt', map_location=device)

# If the state_dict is an ordered dictionary, convert it to a regular dictionary
# if isinstance(state_dict, collections.OrderedDict):
#     state_dict = {k: v for k, v in state_dict.items()}

# print(state_dict.keys())
# Load the state dict into the model
model.load_state_dict(state_dict,strict=False)

print(model.state_dict())
