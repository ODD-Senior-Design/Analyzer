import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
from torch.nn import Module, Linear, ReLU, CrossEntropyLoss

class NeuralNetwork( Module ):

    #TODO: Implement to be dynamic until layer amount and type is chosen
    def __init__( self, input_dimension, hidden_dimension, output_dimension ):
        super( NeuralNetwork, self ).__init__()
        self.__input_layer = Linear( input_dimension, hidden_dimension )
        self.__relu = ReLU()
        self.__output_layer = Linear( hidden_dimension, output_dimension )

    #TODO: Simple propogation of the datapoint throughout the layers, double check that is ideal for this case
    def forward( self, datapoint ) -> Linear:
        datapoint = self.__input_layer( datapoint )
        datapoint = self.__relu( datapoint )
        datapoint = self.__output_layer( datapoint )
        return datapoint

    #TODO: Implement this method and choose correct criterion, `CrossEntropyLoss` is a placeholder
    def train( self, optimizer: Optimizer, criterion: CrossEntropyLoss, num_epochs: int = 1000 ):
        pass

    #TODO: Implement this method properly, this process is just a placeholder
    def predict( self, datapoint: np.ndarray ) -> torch.Tensor:
        datapoint_tensor = torch.from_numpy( datapoint ).float()
        prediction = self.forward( datapoint_tensor )
        return torch.argmax( prediction, dim=1 )
