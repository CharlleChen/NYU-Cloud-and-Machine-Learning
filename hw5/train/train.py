import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

import os

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler



if torch.cuda.is_available():
    device = torch.device('cuda:0')
    pretrained = False
    epoch = 10
    halt = False
else:
    device = torch.device('cpu')
    pretrained = True
    epoch = 1
    halt = True

model = models.mobilenet_v2(pretrained=pretrained)

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_dataset = datasets.ImageFolder('tiny-imagenet-200/train', data_transform)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4,
                                             shuffle=True, num_workers=4)
class_names = image_dataset.classes

def train_model(model, criterion, optimizer, scheduler, num_epochs, halt):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        i = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)


                loss.backward()
                optimizer.step()
            scheduler.step()

            # CPU Training is so slow
            if halt:
                break
            else:
                i += 1

    return model


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, scheduler, epoch, halt)

torch.save(model.state_dict(), '/mnt/mobilenetv2.pth')
