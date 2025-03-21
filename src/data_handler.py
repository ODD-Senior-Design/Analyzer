from roboflow import Roboflow
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from PIL import ImageFile

from typing import List, Dict, Optional, Any
import json

class DataUnpacker():

    def __init__( self, datasets_save_path: str, roboflow_api_key: Optional[str] = None, dataset_manifest: str = '../datasets/dataset_manifests.json' ):
        self.__rf = Roboflow( api_key=roboflow_api_key ) if roboflow_api_key else None
        self.__datasets_save_path = datasets_save_path
        self.__dataset_manifest = dataset_manifest

    def __deserialize_dataset_manifest( self ) -> List[ Dict[ str, Any ] ]:
        with open( self.__dataset_manifest, 'r', encoding='utf-8' ) as f:
            return json.load( f )

    def unpack_datasets( self ) -> Dict[ str, Any ]:
        manifest = self.__deserialize_dataset_manifest()
        datasets = {}
        for dataset_metadata in manifest:

            match dataset_metadata.get( 'provider', '' ):

                case 'roboflow':
                    if not self.__rf:
                        raise ValueError( 'Must define API to use Roboflow as a provider' )

                    workspace_id = dataset_metadata.get( 'workspace_id' )
                    project_id = dataset_metadata.get( 'project_id' )

                    if not workspace_id or not project_id:
                        raise ValueError( 'Workspace ID and Project ID are required for Roboflow' )

                    project = self.__rf.workspace( workspace_id ).project( project_id )
                    version = int( dataset_metadata.get( 'version', 1 ) )
                    dataset = project.version( version )
                    datasets[
                        dataset_metadata.get(
                            'dataset_name',
                            f'my_dataset_{ manifest.index( dataset_metadata ) }',
                        )
                    ] = dataset.images.download( self.__datasets_save_path )

        return datasets

# TODO: Implement class to preprocess datasets for training, validation, etc.
class Preproccesser():

    def __init__( self ) -> None:
        self.__process_flow = transforms.Compose([
            transforms.Resize( 256 ),
            transforms.CenterCrop( 224 ),
            transforms.ToTensor(),
            transforms.Normalize( mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ] )
        ])
        
    def process( self, dataset: Any ) -> torch.Tensor:
        return self.__process_flow( dataset )
    