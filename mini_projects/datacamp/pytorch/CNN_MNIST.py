#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple CNN
"""

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

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
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(10*7*7, 10)

    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 10*7*7)

        # Apply the fully connected layer and return the result
        return self.fc(x)
    


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

net.train()    
    
for i, data in enumerate(trainloader):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)
        
    # Compute the loss function
    loss = criterion(outputs,labels)
        
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()


# Iterate over the data in the test_loader
net.eval()

total = 0
correct = 0

for i, data in enumerate(testloader):
    inputs, labels = data
        
    # Do the forward pass and get the predictions
    outputs = net(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()
    
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total)) 