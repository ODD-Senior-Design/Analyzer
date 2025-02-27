import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
from torch.nn import Module, Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Sequential, AdaptiveAvgPool2d, Dropout

class AlexNet( Module ):

    # From https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
    def __init__( self, num_classes: int = 1000, dropout: float = 0.5 ) -> None:
        super().__init__()
        self.features = Sequential(
            Conv2d( 3, 64, kernel_size=11, stride=4, padding=2 ),
            ReLU( inplace=True ),
            MaxPool2d( kernel_size=3, stride=2 ),
            Conv2d( 64, 192, kernel_size=5, padding=2 ),
            ReLU( inplace=True ),
            MaxPool2d( kernel_size=3, stride=2 ),
            Conv2d( 192, 384, kernel_size=3, padding=1 ),
            ReLU( inplace=True ),
            Conv2d( 384, 256, kernel_size=3, padding=1 ),
            ReLU( inplace=True ),
            Conv2d( 256, 256, kernel_size=3, padding=1 ),
            ReLU( inplace=True ),
            MaxPool2d( kernel_size=3, stride=2 )
        )
        self.avgpool = AdaptiveAvgPool2d( ( 6, 6 ) )
        self.classifier = Sequential(
            Dropout( p=dropout ),
            Linear( 256 * 6 * 6, 4096 ),
            ReLU( inplace=True ),
            Dropout( p=dropout ),
            Linear( 4096, 4096 ),
            ReLU( inplace=True ),
            Linear( 4096, num_classes ),
        )

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        x = self.features( x )
        x = self.avgpool( x )
        x = torch.flatten( x, 1 )
        x = self.classifier( x )
        return x

    #TODO: Implement this method and choose correct criterion, `CrossEntropyLoss` is a placeholder
    def train( self, optimizer: Optimizer, criterion: CrossEntropyLoss, num_epochs: int = 1000 ):
        pass

    #TODO: Implement this method properly, this process is just a placeholder
    def predict( self, datapoint: np.ndarray ) -> torch.Tensor:
        datapoint_tensor = torch.from_numpy( datapoint ).float( )
        prediction = self.forward( datapoint_tensor )
        return torch.argmax( prediction, dim=1 )
