import sys
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoFeatureExtractor, ViTForImageClassification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append('models')
from models.mobilevitv1 import MobileViT
from models.mobilevitv2 import MobileViTv2
from models.mobilevitv3_1 import MobileViTv3_v1
from fvcore.nn import FlopCountAnalysis, flop_count_table 

image_size = (256,256)
num_classes = 10

# Initialize the model

# model = MobileViTv3_v1(
#     image_size=image_size,
#     mode='small',
#     num_classes=num_classes,
#     patch_size=(2,2)
# )

# pretrained = 'pretrained/mobilevitv3_XXS_e300_7098/checkpoint_ema_best.pt'
# state_dict = torch.load(pretrained, map_location=device)
# model.load_state_dict(state_dict)
# model.classifier.fc = nn.Linear(in_features=model.classifier.fc.in_features, out_features=num_classes, bias=True)

# model = models.resnet18(pretrained=True)
# model.fc = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=False),
#     nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
# )

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

# EffNetB0
model = models.efficientnet_b0()
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
)

# #MNASNET0_5
# model = models.mnasnet0_5()
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=True),
#     nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
# )
# model = models.mnasnet0_75()
# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=True),
#     nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)
# )

# MOBILENET_V2
# model = models.mobilenet_v2()
# print(model)
# model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes, bias=True)

# # MOBILENET_V3_SMALL
# model = models.mobilenet_v3_small()
# model.classifier = nn.Sequential(
#     nn.Linear(in_features=model.classifier[0].in_features, out_features=model.classifier[0].out_features, bias=True),
#     nn.Hardswish(),
#     nn.Dropout(p=0.2, inplace=True),
#     nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes, bias=True)
# )


# # MOBILENET_V3_LARGE
# model = models.mobilenet_v3_large()
# model.classifier = nn.Sequential(
#     nn.Linear(in_features=model.classifier[0].in_features, out_features=model.classifier[0].out_features, bias=True),
#     nn.Hardswish(),
#     nn.Dropout(p=0.2, inplace=True),
#     nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes, bias=True)
# )

# # SHUFFLENET_V2_X0_5
# model = models.shufflenet_v2_x1_5()
# model.fc = nn.Sequential(
#     nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True),
# )


# model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')
# model.classifier = nn.Sequential(
#     nn.Linear(in_features=model.classifier.in_features, out_features=num_classes, bias=True),
# )

# import time

# # Define your model and input
# input_data = torch.randn(1, 3, 256, 256)  # Adjust dimensions as per your model's input
# model.eval()

# # Warm-up GPU (if using GPU)
# model(input_data)

# # Number of iterations to measure FPS
# num_iterations = 100

# # Measure FPS
# start_time = time.time()
# for _ in range(num_iterations):
#     # Process frames (simulated by forward pass through the model)
#     with torch.no_grad():
#         model(input_data)  # Change to model(input_data) if using CPU

# end_time = time.time()

# # Calculate FPS
# fps = num_iterations / (end_time - start_time)
# print(f"Frames Per Second (FPS): {fps}")

model.to(device)
x = torch.randn(5, 3, 256, 256).to(device)
flop_analyzer = FlopCountAnalysis(model, x)
print(flop_count_table(flop_analyzer))