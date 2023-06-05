import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import timm
# import report from sklearn
from sklearn.metrics import classification_report

import wandb

import argparse


# Take user args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)

modelName = parser.parse_args().model

wandb.init(project="Stylumia-Internship-Kaggle", name=modelName)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_fp16_reduced_precision_reduction = True

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define image transforms with augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
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
    trainset, batch_size=16, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=16, shuffle=False, num_workers=2)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=range(
    len(dataset.classes)), y=trainset.dataset.targets)
print(class_weights)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Load the pretrained model
#model = torchvision.models.resnet18(pretrained=True)

# Load pretrained model from timm with 7 classes
# Take userinput from the script launch for the model name
model = timm.create_model(
    modelName, pretrained=True, num_classes=7)


# Train, test in each epoch with tqdm progress bar, use functions for train and test


def train(trainloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        

    print("Training loss: ", running_loss/len(trainloader))
    

def test(testloader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions.extend(predicted.tolist())
            ground_truths.extend(labels.tolist())

    print("Testing loss: ", running_loss/len(testloader))
    print("Accuracy: ", correct/total)
    wandb.log({"Accuracy": correct/total})

    # Generate classification report
    report = classification_report(
        ground_truths, predictions, output_dict=True)
    wandb.log({"Classification Report": report})



# Change last fc layer to 7 classes
model.fc = nn.Linear(512, 7)

# Make sure last layer is trainable
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)


# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and test for 10 epochs
print("last Layer Fine Tune (With Class Weights)")
for epoch in range(1,11):
    print("Epoch:", epoch)
    train(trainloader)
    test(testloader)

# Unfreeze all layers and fine-tune with smaller learning rate
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()

# Choose a smaller learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("All Layers Fine Tune (Without Class Weights)")
# Train and test for another 10 epochs
for epoch in range(11,16):
    print("Epoch: ", epoch)
    train(trainloader)
    test(testloader)


# Further reduce the learning rate
optimizer = optim.Adam(model.parameters(), lr=0.00001)

for epoch in range(16,21):
    print("Epoch: ", epoch)
    train(testloader)
    test(trainloader)


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

        submission_df = pd.concat([submission_df, pd.DataFrame(
            {'file_name': [file], 'label': [predicted.item()]})])


submission_df.to_csv(f'{modelName}.csv', index=False)

# Save the trained model
torch.save(model.state_dict(), f'{modelName}.pth')

wandb.close()