import os
import sys
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm
from contextlib import nullcontext

import numpy as np
import random
import torch
from torch.amp import grad_scaler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module, Linear, ReLU, Conv2d, MaxPool2d, Sequential, AdaptiveAvgPool2d, Dropout
from sklearn.metrics import accuracy_score
from typing import List, Optional, Type, Tuple, Dict, Any

class CNN( Module ):

    def __init__( self, model_class: Optional[ Type[ Module ] ] = None, model_kwargs: Optional[ Dict[ str, Any ] ] = None, evaluation_function = torch.sigmoid, device = torch.device( 'cpu' ) ) -> None:
        super().__init__()
        self.__loss_values: List[ float ] = []
        self.__evaluation_function = evaluation_function
        self.__device = device

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

    def train_model( self, dataset: DataLoader, optimizer: Optimizer, loss_fn: Module, num_epochs: int = 10, plot_loss: bool = True, scheduler: Optional[Any] = None, accumulation_steps: int = 1, early_stop_patience: int = 5, seed: int = 42 ) -> None:

        # Reproducibility
        torch.manual_seed( seed )
        np.random.seed( seed )
        random.seed( seed )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        use_amp: bool = self.__device.type == 'cuda'
        amp_context = torch.autocast( self.__device.type ) if use_amp else nullcontext()
        scaler = grad_scaler.GradScaler() if use_amp else None

        if hasattr( self, "compile" ) and use_amp:
            self.compile()

        self.to( self.__device )
        self.train()

        best_loss: float = float('inf')
        patience_counter: int = 0

        for epoch in range( num_epochs ):
            start_time: float = time.time()
            running_loss: float = 0.0

            is_interactive = sys.stdout.isatty()
            dataloader = tqdm( dataset,
                                desc=f"Epoch {epoch+1}/{num_epochs}",
                                dynamic_ncols=not is_interactive,
                                file=sys.stdout if is_interactive else None,
                                disable=False )

            for step, (inputs, labels) in enumerate( dataloader ):
                inputs, labels = inputs.to( self.__device, non_blocking=True ), labels.to( self.__device, non_blocking=True )
                optimizer.zero_grad()

                with amp_context:
                    outputs: torch.Tensor = self( inputs )
                    loss: torch.Tensor = loss_fn( outputs, labels ) / accumulation_steps

                if scaler:
                    scaler.scale( loss ).backward()
                    if ( step + 1 ) % accumulation_steps == 0:
                        scaler.step( optimizer )
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if ( step + 1 ) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item() * accumulation_steps

                if is_interactive:
                    dataloader.set_postfix( loss=loss.item() * accumulation_steps )
                else:
                    print( f"[Epoch { epoch+1 }], Loss: {( loss.item() * accumulation_steps ):.4f}", flush=True )

            if scheduler:
                scheduler.step()

            epoch_loss: float = running_loss / len( dataset )
            epoch_duration: float = time.time() - start_time
            self.__loss_values.append( epoch_loss )
            print( f"Epoch { epoch+1 } Loss: {epoch_loss:.4f} | Duration: {epoch_duration:.2f}s" )

            with open( "training_log.csv", "a", encoding="utf-8" ) as f:
                f.write( f"{ epoch+1 },{epoch_loss:.4f},{epoch_duration:.2f}\n" )

            if plot_loss:
                self.plot_loss( 'Training Loss' )

            # Save model every 5 epochs
            if ( epoch + 1 ) % 5 == 0:
                torch.save( self.state_dict(), f"model_epoch_{ epoch+1 }.pt" )

            # Early stopping
            if epoch_loss > best_loss:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print( f"[INFO] Early stopping at epoch { epoch+1 }", flush=True )
                    break
            else:
                best_loss = epoch_loss
                patience_counter = 0

    def evaluate_model( self, data: DataLoader ) -> float:
        if hasattr( self, "compile" ):
            self.compile()

        self.to( self.__device )
        self.eval()

        all_predictions: List[ float ] = []
        all_labels: List[ float ] = []

        start_time: float = time.time()
        is_interactive = sys.stdout.isatty()
        dataloader = tqdm( data,
                        desc = "Evaluating Model",
                        dynamic_ncols = not is_interactive,
                        file = sys.stdout if is_interactive else None,
                        disable = False )

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to( self.__device ), labels.to( self.__device ).float()
                outputs: torch.Tensor = self( inputs )
                predictions: torch.Tensor = ( self.__evaluation_function( outputs ) > 0.5 ).float()

                all_predictions.extend( predictions.cpu().numpy() )
                all_labels.extend( labels.cpu().numpy() )

        end_time: float = time.time()
        accuracy: float = float( accuracy_score( all_labels, all_predictions ) )
        print( f"Evaluation completed in {end_time - start_time:.2f}s - Accuracy: {accuracy:.4f}" )
        return accuracy

    def get_predictions( self, data: DataLoader ) -> Tuple[ List[ float ], List[ float ] ]:
        self.to( self.__device )
        self.eval()

        all_predictions: List[ float ] = []
        all_labels: List[ float ] = []

        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.to( self.__device ), labels.to( self.__device ).float()
                outputs: torch.Tensor = self( inputs )
                predictions: torch.Tensor = ( self.__evaluation_function( outputs ) > 0.5 ).float()

                all_predictions.extend( predictions.cpu().numpy() )
                all_labels.extend( labels.cpu().numpy() )

        return all_predictions, all_labels

    def test_image( self, image_tensor: torch.Tensor ) -> bool:
        with torch.no_grad():
            output: torch.Tensor = self( image_tensor )
            return self.__evaluation_function( output ).float() > 0.5

    def save_model( self, path: str ) -> None:
        if not path:
            timestamp = datetime.datetime.now().strftime( "%Y%m%d_%H%M%S" )
            path = f"./saved_models/{ self.model.__class__.__name__ }/model_{ timestamp }.pt"

        os.makedirs( os.path.dirname( path ), exist_ok=True )
        torch.save( self.state_dict(), path )
        print( f"Model saved to { path }" )

    def load_model( self, model_path: str ) -> None:

        if not model_path.endswith( '.pth' ):
            raise ValueError( 'Model file must be a PyTorch (.pth) file' )

        if not os.path.exists( model_path ):
            raise FileNotFoundError( f'Model file not found at { model_path }' )

        self.load_state_dict( torch.load( model_path, map_location=self.__device ) )
        self.to( self.__device )
        self.eval()
        print( f"Model loaded from { model_path }" )
