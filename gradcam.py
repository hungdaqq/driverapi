import sys
sys.path.append('models')
from models.mobilevitv1 import MobileViT
from models.MyModel import MyModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # Import the ClassifierOutputTarget class
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='Process image with GradCAM')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
parser.add_argument('--model', type=str, required=True, help='Path to the pretrained model')

args = parser.parse_args()

image_path = args.image_path
pretrained = args.model

num_classes = 16

if pretrained == 'MyModel':
    image_size = (256,256)
    model = MyModel(
        image_size=image_size,
        mode='x_small',
        num_classes=num_classes,
        patch_size=(2,2)
    )
    pretrained = 'pretrained/MyModel/mymodel_0.9038.pt'
    model.load_state_dict(torch.load(pretrained, map_location=torch.device('cpu')))
    model.eval()
    # Define target layer(s) for GradCAM
    target_layers = [model.conv_1x1_exp]  # Using the last layer of the final ResNet block
elif pretrained == 'resnet50':
    image_size = (224,224)
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
    )
    pretrained = 'pretrained/resnet50/mymodel_0.9416.pt'
    model.load_state_dict(torch.load(pretrained, map_location=torch.device('cpu')))
    model.eval()
    target_layers = [model.layer4[-1]]  # Using the last layer of the final ResNet block

elif pretrained == 'vit':
    image_size = (224,224)
    import timm
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=False, num_classes=num_classes)
    pretrained = 'pretrained/ViT/mymodel_0.8305.pt'
    model.load_state_dict(torch.load(pretrained, map_location=torch.device('cpu')))
    model.eval()
    target_layers = [model.blocks[-1].norm1] 

# Define transformations to apply to the image
transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the image to match the input size expected by the model
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# Load the image using PIL (Python Imaging Library)
image = Image.open(image_path).convert('RGB')

input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

targets = [ClassifierOutputTarget(i) for i in range(16)]  # Assuming you want to visualize class indices from 0 to 15

import os 
save_dir = 'results/' + pretrained.split('/')[1] + '/' + image_path.split('/')[3] + '/'
os.makedirs(save_dir, exist_ok = True) 

for i, target in enumerate(targets):
    grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
    grayscale_cam = cv2.resize(grayscale_cam[0], (image.width, image.height))
    visualization = show_cam_on_image(np.array(image) / 255.0, grayscale_cam, use_rgb=True)
    save_image = save_dir + str(i) + '.png'
    print(save_image)
    cv2.imwrite(save_image, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))