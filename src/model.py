import os
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm
from contextlib import nullcontext

import torch
from torch.amp import grad_scaler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module, Linear, ReLU, Conv2d, MaxPool2d, Sequential, AdaptiveAvgPool2d, Dropout
from sklearn.metrics import accuracy_score
from typing import List, Optional, Type, Dict, Any

from sample_model import BinaryAlexNet

class CNN( Module ):

    def __init__( self, model_class: Optional[ Type[ Module ] ] = None, model_kwargs: Optional[ Dict[ str, Any ] ] = None, evaluation_function = torch.sigmoid ) -> None:
        super().__init__()
        self.__loss_values: List[ float ] = []
        self.__evaluation_function = evaluation_function

        if model_class is None:
            model_class = self.BinaryAlexNet

        if model_kwargs is None:
            model_kwargs = {}

        self.model: Module = model_class( **model_kwargs )

    class BinaryAlexNet( Module ):
        #* From https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        #* Modified to be for binary classification
        def __init__( self, dropout: float = 0.5, num_classes = 2 ) -> None:
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
                Linear( 4096, num_classes ),
            )

        def forward( self, x: torch.Tensor ) -> torch.Tensor:
            x = self.__features( x )
            x = self.__avgpool( x )
            x = torch.flatten( x, 1 )
            x = self.__classifier( x )
            return x

    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        return self.model( x )

    def plot_loss( self, title: str = 'Model Loss' ) -> None:
        plt.plot( self.__loss_values, label=title )
        plt.xlabel( 'Epochs' )
        plt.ylabel( 'Loss' )
        plt.show()

    def train_model( self, dataset: DataLoader, optimizer: Optimizer, loss_fn: Module, num_epochs: int = 10, compute_device: torch.device = torch.device( 'cpu' ), plot_loss: bool = True ) -> None:

        device_type: str = compute_device.type
        use_amp: bool = device_type == 'cuda'
        amp_context = torch.autocast( device_type = device_type ) if use_amp else nullcontext()
        scaler = grad_scaler.GradScaler() if use_amp else None

        if hasattr( self, "compile" ) and use_amp:
            self.compile()

        self.to( compute_device )
        self.train()

        for epoch in range( num_epochs ):
            start_time: float = time.time()
            running_loss: float = 0.0

            dataloader = tqdm( dataset, desc=f"Epoch { epoch+1 }/{ num_epochs }" )

            for inputs, labels in dataloader:
                inputs, labels = inputs.to( compute_device ), labels.to( compute_device )
                optimizer.zero_grad()

                with amp_context:
                    outputs: torch.Tensor = self( inputs )
                    loss: torch.Tensor = loss_fn( outputs, labels )

                if scaler:
                    scaler.scale( loss ).backward()
                    scaler.step( optimizer )
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                dataloader.set_postfix( loss=loss.item() )

            epoch_loss: float = running_loss / len( dataset )
            epoch_duration: float = time.time() - start_time
            self.__loss_values.append( epoch_loss )
            print( f"Epoch { epoch+1 } Loss: { epoch_loss } | Duration: { epoch_duration }s" )

            if plot_loss:
                self.plot_loss( 'Training Loss' )

    def evaluate_model( self, data: DataLoader, compute_device: torch.device = torch.device( 'cpu' ) ) -> float:
        if hasattr( self, "compile" ):
            self.compile()

        self.to( compute_device )
        self.eval()

        all_predictions: List[ float ] = []
        all_labels: List[ float ] = []

        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.to( compute_device ), labels.to( compute_device ).float()
                outputs: torch.Tensor = self( inputs )
                predictions: torch.Tensor = ( self.__evaluation_function( outputs ) > 0.5 ).float()
                all_predictions.extend( predictions.cpu().numpy() )
                all_labels.extend( labels.cpu().numpy() )

        accuracy: float = accuracy_score( all_labels, all_predictions )
        print( f"Test Accuracy: { accuracy }" )
        return accuracy

    def test_image( self, image_tensor: torch.Tensor ) -> bool:
        with torch.no_grad():
            output: torch.Tensor = self( image_tensor )
            return ( self.__evaluation_function( output ).float() > 0.5 )

    def save_model( self, path: str ) -> None:
        if path is None:
            timestamp = datetime.datetime.now().strftime( "%Y%m%d_%H%M%S" )
            path = f"./saved_models/{ BinaryAlexNet.__name__ }/model_{ timestamp }.pt"

        os.makedirs( os.path.dirname( path ), exist_ok=True )
        torch.save( self.state_dict(), path )
        print( f"Model saved to { path }" )

    def load_model( self, model_path: str, device: torch.device = torch.device( 'cpu' ) ) -> None:

        if not model_path.endswith( '.pth' ):
            raise ValueError( 'Model file must be a PyTorch (.pth) file' )

        if not os.path.exists( model_path ):
            raise FileNotFoundError( f'Model file not found at { model_path }' )

        self.load_state_dict( torch.load( model_path, map_location=device ) )
        self.to( device )
        self.eval()
        print( f"Model loaded from { model_path }" )
