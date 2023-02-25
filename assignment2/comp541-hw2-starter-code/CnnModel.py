import torch
import torch.nn as nn




class SimpleCNN(torch.nn.Module):
    def __init__(self,use_cuda=False, pooling=False, use_dropout=False, dropout_prob=0.5):
        super(SimpleCNN, self).__init__()
        self.use_cuda = use_cuda
        self.pooling = pooling
        self.use_dropout = use_dropout
        self.conv_layer1 =  torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.pool_layer1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)
        self.pool_layer2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        if pooling:
            self.fully_connected_layer = nn.Linear(64, 64)
            self.final_layer = nn.Linear(64, 11)
        else:
            self.fully_connected_layer = nn.Linear(1600, 64)
            self.final_layer = nn.Linear(64, 11)

        # Add dropout Layers
        if use_dropout == True:
            self.dropout_layer1 = torch.nn.Dropout(dropout_prob)
            self.dropout_layer2 = torch.nn.Dropout(dropout_prob)

        self.conv_layer3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2)

    def forward(self,inp):
        x = torch.nn.functional.relu(self.conv_layer1(inp))
        if self.pooling:
            x = self.pool_layer1(x)
        x = torch.nn.functional.relu(self.conv_layer2(x))
        if self.pooling:
            x = self.pool_layer2(x)
        x = x.reshape(x.size(0), -1)
        if self.use_dropout:
            x = self.dropout_layer1(x)
        x = torch.nn.functional.relu(self.fully_connected_layer(x))
        if self.use_dropout:
            x = self.dropout_layer2(x)
        x = self.final_layer(x)
        return x
