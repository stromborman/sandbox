#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN with L2 regularization, Dropouts, Batch Normalizating, and Epochs (set up for early stopping)
"""


import torch
from torchvision import datasets
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=5, eps=1e-05, momentum=0.5),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2), 
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=10, eps=1e-05, momentum=0.5),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=20, eps=1e-05, momentum=0.5),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2), 
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=0.5)
                                      )
        self.classifier = nn.Sequential(
                            nn.Dropout(p=.3),
                            nn.Linear(7 * 7 * 40, 1024), 
                            nn.ReLU(inplace=True),
                            nn.BatchNorm1d(num_features=1024, eps=1e-05, momentum=0.5),
                            nn.Dropout(p=.3),
                            nn.Linear(1024, 2048), 
                            nn.ReLU(inplace=True),
                            nn.BatchNorm1d(num_features=2048, eps=1e-05, momentum=0.5),
                            nn.Dropout(p=.3),
                            nn.Linear(2048, 10))
        
    def forward(self, x):
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 40)
        
        # Classify the images
        x = self.classifier(x)
        return x
    


# Shuffle the indices
indices = np.arange(60000)
np.random.shuffle(indices)

# Build the train loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('~/anaconda3/share/mnist', download=False, train=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                     batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:45000]))

# Build the validation loader
val_loader = torch.utils.data.DataLoader(datasets.MNIST('~/anaconda3/share/mnist', download=False, train=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[45000:55000]))

# Build the test loader
test_loader = torch.utils.data.DataLoader(datasets.MNIST('~/anaconda3/share/mnist', download=False, train=False,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[55000:]))

# net = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=.0001)

# net.train()

# for i, data in enumerate(train_loader):
#     inputs, labels = data
#     optimizer.zero_grad()

#     # Compute the forward pass
#     outputs = net(inputs)
        
#     # Compute the loss function
#     loss = criterion(outputs,labels)
        
#     # Compute the gradients
#     loss.backward()
    
#     # Update the weights
#     optimizer.step()

# total = 0
# correct = 0

# for i, data in enumerate(val_loader):
#     inputs, labels = data
        
#     # Do the forward pass and get the predictions
#     outputs = net(inputs)
#     _, outputs = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (outputs == labels).sum().item()
    
# print('The validation set accuracy of the network is: %d %%' % (100 * correct / total)) 

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=.0001)


train_data_size = 45000
val_data_size = 10000

epochs = 30
train_loss = [] 
val_loss = []
t_accuracy_gain = []
accuracy_gain = []

for epoch in range(epochs):
   
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    total_t = 0
    # training our model
    for i, (image, label) in enumerate(train_loader):

        # image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        pred_t = model(image)

        loss = criterion(pred_t, label)
        total_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        pred_t = torch.nn.functional.softmax(pred_t, dim=1)
        for i, p in enumerate(pred_t):
            if label[i] == torch.max(p.data, 0)[1]:
                total_t = total_t + 1
                
    accuracy_t = total_t / train_data_size
    t_accuracy_gain.append(accuracy_t)



    total_train_loss = total_train_loss / (i + 1)
    train_loss.append(total_train_loss)
    
    # validating our model
    total = 0
    for i, (image, label) in enumerate(val_loader):
        pred = model(image)
        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy = total / val_data_size
    accuracy_gain.append(accuracy)

    total_val_loss = total_val_loss / (i + 1)
    val_loss.append(total_val_loss)

    #if epoch % 5 == 0:
    print('\nEpoch: {}/{}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch, epochs, total_train_loss, total_val_loss, accuracy))






