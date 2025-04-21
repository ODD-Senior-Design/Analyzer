from warnings import warn
from roboflow import Roboflow
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from glob import glob
from PIL import Image
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from os import mkdir, path, listdir, remove
from shutil import copy as copy_file
from typing import List, Dict, Optional, Any
import json
import yaml

# TODO: Implement class to preprocess datasets for training, validation, etc.
class Preproccessor():

    def __init__( self ) -> None:
        self.__process_flow = transforms.Compose([
            transforms.Resize( 256 ),
            transforms.CenterCrop( 224 ),
            transforms.ToTensor(),
            transforms.Normalize( mean=[ 0.485, 0.456, 0.406 ], std=[ 0.229, 0.224, 0.225 ] )
        ])

    def process( self, dataset: Any ) -> torch.Tensor:
        return self.__process_flow( dataset )

    def get_transform( self ) -> transforms.Compose:
        return self.__process_flow

class DataUnpacker():

    def __init__( self, datasets_save_path: str, roboflow_api_key: Optional[ str ] = None ):
        self.__rf = Roboflow( api_key=roboflow_api_key, model_format='yolov5' ) if roboflow_api_key else None
        self.__datasets_save_path = datasets_save_path
        self.__dataset_manifest = f'{ datasets_save_path }/datasets_manifest.json'

    def __deserialize_dataset_manifest( self ) -> List[ Dict[ str, Any ] ]:
        if not path.exists( self.__dataset_manifest ):
            raise FileNotFoundError( f'Datasets manifest not found at path: { self.__dataset_manifest }' )

        with open( self.__dataset_manifest, 'r', encoding='utf-8' ) as f:
            return json.load( f )

    def __get_class_indices( self, data_yaml_path: str, labels: Optional[ list[str] ] = None ) -> Dict[ str, int ]:
        with open( data_yaml_path, 'r', encoding='utf-8' ) as f:
            data = yaml.safe_load( f )

        class_names = data.get( 'names', [] )
        return (
            { label: class_names.index( label ) for label in labels if label in class_names } if labels
            else { cl: class_names.index( cl ) for cl in class_names }
        )

    def __filter_worker( self, root_path: str, class_indicies: set[ int ], in_place: bool ):
        labels_dir = f'{ root_path }/labels'
        images_dir = f'{ root_path }/images'
        filtered_labels_dir = labels_dir if in_place else f'{ path.basename( root_path ) }/labels'
        filtered_images_dir = images_dir if in_place else f'{ path.basename( root_path ) }/images'
        delete_labels = set()
        delete_images = set()

        if not in_place:
            mkdir( filtered_labels_dir )
            mkdir( filtered_images_dir )

        for label_file in glob( f'{ labels_dir }/*.txt' ):
            with open( label_file, 'r', encoding='utf-8' ) as f:
                file_lines = f.readlines()

            if all( int( line.split()[0]) not in class_indicies for line in file_lines if line.strip() ):
                delete_labels.add( label_file )
                continue

            if not in_place:
                copy_file( label_file, f'{ filtered_labels_dir }/{ label_file }' )

            base_name = label_file.split('.')[0]

            for ext in [ '.jpg', 'jpeg', '.png' ]:
                image_file = f'{ images_dir }/{ base_name }{ ext }'
                if path.exists( image_file ):
                    if not in_place:
                        copy_file( image_file, f'{ filtered_images_dir }/{ image_file }' )
                else:
                    delete_images.add( image_file )

            if in_place:
                for image_file, label_file in zip( delete_images, delete_labels ):
                    remove( image_file )
                    remove( label_file )


    def __filter( self, dataset_path: str, labels: List, in_place: bool = False ) -> None:
        if not path.exists( f'{ dataset_path }/data.yaml' ):
            raise FileNotFoundError( f"There is no 'data.yaml' in '{ dataset_path }', please have dataset in YOLO format." )

        class_indicies = self.__get_class_indices( f'{ dataset_path }/data.yaml', labels )
        print( f'Got matching indicies for: { class_indicies }' )
        if len( class_indicies ) != len( labels ):
            raise ValueError( f'Not all labels were found....perhaps a typo? Matched Labels:\n{ class_indicies }\n\nAvailable labels:\n{ self.__get_class_indices( f'{ dataset_path }/data.yaml' ) }' )

        class_indicies = set( class_indicies.values() )

        if not path.exists( f'{ dataset_path }/train' ) or not path.exists( f'{ dataset_path }/valid' ) or not path.exists( f'{ dataset_path }/test' ):
            raise FileNotFoundError( "Please have the dataset in YOLO format with a 'train','valid', and 'test' directory under the root directory of the datset." )

        splits = [ 'train', 'valid', 'test' ]

        with ThreadPoolExecutor( max_workers=3 ) as executor:

            futures = [
                executor.submit( self.__filter_worker, f'{ dataset_path }/{ split }', class_indicies, in_place ) for split in splits
            ]

            for future in futures:
                future.result()

    def unpack_datasets( self ) -> List[ str ]:
        manifest = self.__deserialize_dataset_manifest()

        dataset_paths = []

        for dataset_metadata in manifest:

            filter_labels: Optional[ List ] = dataset_metadata.get( 'filter_labels' )
            filter_in_place: bool = dataset_metadata.get( 'filter_in_place', 1 ) == 1

            match dataset_metadata.get( 'provider' ):

                case 'roboflow':
                    if not self.__rf:
                        raise ValueError( 'Must define API Key to use Roboflow as a provider' )

                    workspace_id = dataset_metadata.get( 'workspace_id' )
                    project_id = dataset_metadata.get( 'project_id' )

                    if not workspace_id or not project_id:
                        raise ValueError( 'Workspace ID and Project ID are required for Roboflow' )

                    project = self.__rf.workspace( workspace_id ).project( project_id )
                    version = int( dataset_metadata.get( 'version', 1 ) )
                    dataset = project.version( version )
                    dataset.download( location=self.__datasets_save_path )

                    dataset_path = f'{ self.__datasets_save_path }/{ project_id }'

                    if filter_labels:
                        self.__filter( dataset_path, filter_labels, filter_in_place )

                    dataset_paths.append( dataset_path )

                case 'local':
                    dataset_path = dataset_metadata.get( 'dataset_path' )

                    if not dataset_path or not path.exists( dataset_path ):
                        raise FileNotFoundError( f'Dataset not found at { dataset_path }' )

                    dataset_paths.append( dataset_path )

                case _:
                    warn( "'Provider' was either not defined or not supported... Currently the only supported providers are 'roboflow' and 'local'" )

        return dataset_paths

    class CombinedDataset( torch.utils.data.Dataset ):
        def __init__( self, combined_dataset_path: str , transform: Any = None) -> None:
            self.data_dir = combined_dataset_path
            self.images = listdir( self.data_dir )
            self.transform = transform

        def __len__( self ) -> int:
            return len( self.images )

        def __getitem__( self, index ) -> np.ndarray:
            image_path = path.join( self.data_dir, self.images[ index ] )
            image = np.array( Image.open( image_path ) )

            if self.transform:
                image = self.transform( image )

            return image

    def get_combined_dataset_dataloader( self, combined_dataset_path: str, preprocess=True, batch_size: int = 32, shuffle: bool = True, num_workers: int = 2 ) -> DataLoader:
        transform = Preproccessor().get_transform() if preprocess else None
        combined_dataset = self.CombinedDataset( combined_dataset_path, transform )
        return DataLoader( dataset=combined_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers )
