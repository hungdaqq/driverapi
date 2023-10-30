import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from mobilevitv1 import MobileViT
torch.manual_seed(0)

image_size = (256,256)
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your model architecture (make sure it matches the one used during training)
model = MobileViT(image_size, 'xx_small', 10)
model = model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize(image_size),  # Adjust the size as needed
    transforms.ToTensor(),
])

# Define the path to your dataset
dataset_path = '/home/hung/statefarm/imgs/train/'

# Create the ImageFolder dataset
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, '
              f'Train Loss: {total_loss:.4f}, Train Acc: {total_correct}', end='\r')

    train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    val_loss = 0

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            print(f'Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(val_loader)}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {total_correct}', end='\r')


    val_loss /= len(val_loader)
    val_accuracy = total_correct / total_samples

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


# Save the trained model
torch.save(model.state_dict(), 'test.pt')