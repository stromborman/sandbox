#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple NN
"""

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

relu = nn.ReLU()


transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((.1307), ((.3081)))])

trainset = torchvision.datasets.MNIST('~/anaconda3/share/mnist', train=True, \
                                      download=False, transform=transform)
testset = torchvision.datasets.MNIST('~/anaconda3/share/mnist', train = False, \
                               download=False, transform=transform)
    
# Prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0) 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(28*28,400)
        self.layer2 = nn.Linear(400,400)
        self.layer3 = nn.Linear(400,400)
        self.layer4 = nn.Linear(400,400)
        self.layer5 = nn.Linear(400,10)

        
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()   
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
  
for batch_idx, data_target in enumerate(trainloader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output,target)
    loss.backward()
    optimizer.step()


# Set the model in eval mode
model.eval()

total = 0
correct = 0

for i, data in enumerate(testloader):
    inputs, labels = data
    
    # Put each image into a vector
    inputs = inputs.view(-1, 28*28)
    
    # Do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()
    
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))       