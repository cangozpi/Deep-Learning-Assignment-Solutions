import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, input):
        return self.model(input)


class AlexNetExtension(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # Load pre-trained AlexNet
        alexnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.alexnetBackbone_features = alexnet_model.features
        self.alexnetBackbone_avgpool = alexnet_model.avgpool
        self.freeze_alexnet_backbone()
        
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(), # [256*6*6]
            torch.nn.Linear(256*6*6, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, output_size)
        )

    def freeze_alexnet_backbone(self):
        # Freeze pre-trained AlexNet Backbone
        for param in self.alexnetBackbone_features.parameters():
            param.requires_grad = False
        for param in self.alexnetBackbone_avgpool.parameters():
            param.requires_grad = False
        self.alexnetBackbone_features.eval()
        self.alexnetBackbone_avgpool.eval()
        
    def forward(self, input):
        with torch.no_grad():
            out = self.alexnetBackbone_features(input) # [256, 6, 6]
            out = self.alexnetBackbone_avgpool(out) # [256, 6, 6]
        out = self.model(out)
        
        return out