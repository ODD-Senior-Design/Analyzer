import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module, Linear, ReLU, Conv2d, MaxPool2d, Sequential, AdaptiveAvgPool2d, Dropout
from sklearn.metrics import accuracy_score
from typing import List

class BinaryAlexNet( Module ):
    #* From https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
    #* Modified to be for binary classification
    def __init__( self, dropout: float = 0.5 ) -> None:
        super().__init__()
        self.__features = Sequential(
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
            MaxPool2d( kernel_size=3, stride=2 ),
      )
        self.__avgpool = AdaptiveAvgPool2d( ( 6, 6 ) )
        self.__classifier = Sequential(
            Dropout( p=dropout ),
            Linear( 256 * 6 * 6, 4096 ),
            ReLU( inplace=True ),
            Dropout( p=dropout ),
            Linear( 4096, 4096 ),
            ReLU( inplace=True ),
            Linear( 4096, 1 ),  # 1 for Binary classification, otherwise should be equal to the number of classes
      )
        self.__loss_values: List[ float ] = []
        self.__evaluation_function = torch.sigmoid

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        x = self.__features( x )
        x = self.__avgpool( x )
        x = torch.flatten( x, 1 )
        x = self.__classifier( x )
        return x

    def plot_loss( self, title: str = 'Model Loss' ) -> None:
        plt.plot( self.__loss_values, label=title )
        plt.xlabel( 'Epochs' )
        plt.ylabel( 'Loss' )
        plt.show()

    def train_model( self, data: DataLoader, optimizer: Optimizer, criterion: Module, num_epochs: int = 10, compute_device: torch.device = torch.device( torch.device( 'cpu' ) ), plot_loss: bool = True ) -> None:
        self.to( compute_device )
        self.train()

        for epoch in range( num_epochs ):
            running_loss: float = 0.0

            for inputs, labels in data:
                inputs, labels = inputs.to( compute_device ), labels.to( compute_device )
                optimizer.zero_grad()
                outputs: BinaryAlexNet = self( inputs )
                loss: Module = criterion( outputs, labels )
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.__loss_values.append( running_loss / len( data ) )
            print( f'Epoch: { epoch + 1 } / { num_epochs }, Epoch Loss: { running_loss / len( data ):.4f }' )

            if plot_loss:
                self.plot_loss( 'Training Loss' )

    def evaluate_model( self, data: DataLoader, compute_device: torch.device = torch.device( 'cpu' ) ) -> torch.Tensor:
        self.to( compute_device )
        self.eval()
        all_predictions: List[ float ] = []
        all_labels: List[ float ] = []

        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.to( compute_device ), labels.to( compute_device ).float()
                outputs: BinaryAlexNet = self( inputs )
                predictions: torch.Tensor = ( self.__evaluation_function( outputs ) > 0.5 ).float()
                all_predictions.extend( predictions.cpu().numpy() )
                all_labels.extend( labels.cpu().numpy() )

        accuracy: float = accuracy_score( all_labels, all_predictions )
        print( f"Test Accuracy: { accuracy:.4f }" )
        return accuracy

    def save_model( self, path: str ) -> None:
        torch.save( self.state_dict(), path )
        print( f"Model saved to { path }" )

    def load_model( self, path: str, device: torch.device = torch.device( 'cpu' ) ) -> None:
        self.load_state_dict( torch.load( path, map_location=device ) )
        self.to( device )
        self.eval()
        print( f"Model loaded from { path }" )
