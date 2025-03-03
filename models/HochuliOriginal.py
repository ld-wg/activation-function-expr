import torch
import torch.nn as nn
import torch.nn.functional as F

class HochuliOriginal(nn.Module):
    def __init__(self):
        super(HochuliOriginal, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # max pooling layer 2x2 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 32x32 -> 16x16 -> 8x8 -> 4x4.
        # 64 channels * 4 * 4 = 1024.
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        # Final classification layer: mapping 64 features to 2 classes.
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        # layer 1: Convolution -> ReLU -> Max Pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # layer 2: Convolution -> ReLU -> Max Pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # layer 3: Convolution -> ReLU -> Max Pooling
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # flatten the features for the dense layers
        x = x.view(x.size(0), -1)  # torch.flatten(x, 1)
        
        # dense layer with 64 units followed by ReLU
        x = F.relu(self.fc1(x))
        
        # final layer for classification (softmax applied to output probabilities)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x