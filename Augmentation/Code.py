import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_fp16_reduced_precision_reduction = True


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=7):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define image transforms with augmentations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.RandomRotation(20),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(
        0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define the dataset
dataset = torchvision.datasets.ImageFolder(
    root='/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/Modified/', transform=transform)

# Split the dataset into train and test sets
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

# Define data loaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)

# Calculate class weights
# class_weights = compute_class_weight('balanced', classes=range(
#     len(dataset.classes)), y=trainset.dataset.targets)
# print(class_weights)
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Load the pretrained model
model = torchvision.models.resnet18(pretrained=True)

# Define the loss function with class weights
criterion = SCELoss(0.1, 1)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop


def train():
    model.train()
    running_loss = 0.0
    train_bar = tqdm(trainloader, desc='Training', leave=False)
    for i, data in enumerate(train_bar, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.set_postfix({'Loss': running_loss / (i + 1)})
    train_bar.close()
    print("Training loss:", running_loss / len(trainloader))

# Testing loop


def test():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    test_bar = tqdm(testloader, desc='Testing', leave=False)
    with torch.no_grad():
        for i, data in enumerate(test_bar, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_bar.set_postfix(
                {'Loss': running_loss / (i + 1), 'Accuracy': correct / total})
    test_bar.close()
    print("Testing loss:", running_loss / len(testloader))
    print("Accuracy:", correct / total)


# Change last fc layer to 7 classes
model.fc = nn.Linear(512, 7)

model = model.to(device)


# Train and test for 10 epochs
for epoch in range(10):
    print("Epoch:", epoch)
    train()
    test()

# Unfreeze all layers and fine-tune with smaller learning rate
for param in model.parameters():
    param.requires_grad = True

# Choose a smaller learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train and test for another 10 epochs
for epoch in range(10):
    print("Epoch:", epoch)
    train()
    test()

# Further reduce the learning rate
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Train and test for another 10 epochs
for epoch in range(10):
    print("Epoch:", epoch)
    train()
    test()

# Make predictions with the model on the test directory and export to CSV
submission_df = pd.DataFrame(columns=['file_name', 'label'])
test_dir = '/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/test'

model.eval()
with torch.no_grad():
    for file in os.listdir(test_dir):
        img = Image.open(os.path.join(test_dir, file))
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        output = model(img)
        _, predicted = torch.max(output.data, 1)

        submission_df = submission_df.append({
            'file_name': file,
            'label': predicted.item()
        }, ignore_index=True)


submission_df.to_csv('SCELoss_Augem.csv', index=False)

# Save the trained model
torch.save(model.state_dict(), 'SCELoss_Augem.pth')
