from warnings import warn
from dotenv import load_dotenv
from roboflow import Roboflow
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from glob import glob
from PIL import Image
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from os import getenv, mkdir, path, listdir, remove
from shutil import copy as copy_file, rmtree
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
        self.__rf = Roboflow( api_key=roboflow_api_key, model_format='folder' ) if roboflow_api_key else None
        self.__datasets_save_path = datasets_save_path
        self.__dataset_manifest = f'{ datasets_save_path }/datasets_manifest.json'
        self.__remove_list_path = f'{ datasets_save_path }/remove_list.csv'

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

    def __filter_worker_yolo( self, root_path: str, class_labels: List[ str ], in_place: bool ):
        class_indicies = self.__get_class_indices( f'{ root_path }/../data.yaml', class_labels )
        if len( class_indicies ) != len( class_labels ):
            raise ValueError( f'Not all labels were found....perhaps a typo?\nRequested Labels:\n{ class_labels }\n\nMatched Labels:\n{ class_indicies }\n\nAvailable labels:\n{ self.__get_class_indices( f'{ root_path }/../data.yaml' ) }' )

        class_indicies = set( class_indicies.values() )
        labels_dir = f'{ root_path }/labels'
        images_dir = f'{ root_path }/images'
        filtered_labels_dir = labels_dir if in_place else f'{ path.basename( root_path ) }/filtered_labels'
        filtered_images_dir = images_dir if in_place else f'{ path.basename( root_path ) }/filtered_images'
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
                    if path.exists( image_file ):
                        remove( image_file )
                    if path.exists( label_file ):
                        remove( label_file )

    def __filter_worker_folder( self, root_path: str, class_labels: List[ str ], in_place: bool = True ) -> None:

        if not in_place:
            warn( 'Images must be filtered in place for this format', UserWarning )

        if any( label not in listdir( root_path ) for label in class_labels ):
            matched_labels = [ label for label in class_labels if label in listdir( root_path ) ]
            raise ValueError( f'Not all labels were found....perhaps a typo?\nRequested Labels:\n{ class_labels }\n\nMatched Labels:\n{ matched_labels }\n\nAvailable labels:\n{ class_labels }' )

        dirs = { name for name in listdir( root_path ) if path.isdir(f'{ root_path }/{ name }') }

        dirs.difference_update( class_labels )

        for label_dir in dirs:
            if path.exists( f'{ root_path }/{ label_dir }' ):
                rmtree( f'{ root_path }/{ label_dir }' )

    def __filter( self, dataset_path: str, labels: List, in_place: bool = False, dataset_format = 'folder' ) -> None:
        if not path.exists( f'{ dataset_path }/data.yaml' ) and 'yolo' in dataset_format:
            warn( f"There is no 'data.yaml' in '{ dataset_path }', dataset not in YOLOv5 format, assuming folder format", )
            dataset_format = 'folder'

        splits = [ 'train', 'test' ]

        if not path.exists( f'{ dataset_path }/train' ):
            raise FileNotFoundError( "Please have the dataset in YOLO format with atleast a 'train' and 'test' directory under the root directory of the datset. Optionally include a 'valid' directory." )

        if not path.exists( f'{ dataset_path }/test' ):
            warn( "No 'test' directory found. Highly reccomended to include one. Will continue to only filter 'train' directory" )
            splits.pop()

        if path.exists( f'{ dataset_path }/valid' ):
            splits.append( 'valid' )

        worker = self.__filter_worker_yolo if 'yolo' in dataset_format else self.__filter_worker_folder

        with ThreadPoolExecutor( max_workers=3 ) as executor:

            futures = [ executor.submit( worker, f'{ dataset_path }/{ split }', labels, in_place ) for split in splits ]

            for future in futures:
                future.result()

    def __clean_worker( self, image_path: str ) -> Optional[ str ]:
        image_path = f'{ self.__datasets_save_path }/{ image_path }'
        try:
            remove( image_path )
            return None
        except FileNotFoundError:
            return f"File not found: { image_path }"
        except Exception as e:
            return f"Error deleting { image_path }: { e }" #! 433_1_jpg.rf.ec625197216f206783d6c12e2fca079f (Test if deleted)

    def __clean( self, remove_list: str, max_workers: int = 8 ) -> None:
        removal_df = pd.read_csv( remove_list )
        file_paths: list[str] = removal_df['filepath'].dropna().tolist()

        with ThreadPoolExecutor( max_workers=max_workers ) as executor:
            results = list( executor.map( self.__clean_worker, file_paths ) )

        for result in results:
            if result:
                warn( result, UserWarning )


    def unpack_datasets( self, overwrite: bool = True, clean: bool =  True, clean_max_workers: int = 8 ) -> List[ str ]:
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

                    print( f'Loaded { workspace_id }-{ project_id }-{ version }' )

                    dataset_path = f'{ path.abspath( self.__datasets_save_path ) }/{ project_id }-{ version }'
                    dataset_format: Optional[ str ] = dataset_metadata.get( 'format' )
                    dataset.download( location = dataset_path, model_format=dataset_format, overwrite=dataset_metadata.get( 'overwrite', overwrite ) == 1 and overwrite )

                    if filter_labels:
                        self.__filter( dataset_path, filter_labels, filter_in_place, dataset_format or 'folder' )
                        print( 'Filtered Dataset\n' )

                    dataset_paths.append( dataset_path )

                case 'local':
                    dataset_path = dataset_metadata.get( 'dataset_path' )

                    if not dataset_path or not path.exists( dataset_path ):
                        raise FileNotFoundError( f'Dataset not found at { dataset_path }' )

                    dataset_paths.append( dataset_path )

                case _:
                    warn( "'Provider' was either not defined or not supported... Currently the only supported providers are 'roboflow' and 'local'", UserWarning )

        if clean:
            if not path.exists( self.__remove_list_path ):
                raise ValueError( "Must define a 'remove_list.csv' file in the datasets path in order to use cleaning function" )
            self.__clean( self.__remove_list_path, clean_max_workers )
            print( 'Cleaned datasets' )

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

def unpack( datasets_path: str, roboflow_api_key: Optional[ str ] = None, overwrite: bool = True, clean: bool = True, clean_max_workers: int = 8 ) -> None:
    if not roboflow_api_key:
        warn( 'Environment variable "ROBOFLOW_API_KEY" is not set. Keep in mind using Roboflow as a provider is not possible then.', UserWarning )

    dataset_unpacker = DataUnpacker( datasets_save_path=datasets_path, roboflow_api_key=roboflow_api_key )
    dataset_unpacker.unpack_datasets( overwrite, clean, clean_max_workers )

    print( 'Datasets unpacked successfully! Please manually verify and combine datasets into a "combined_datasets" directory before training' )
    exit( 0 )

if __name__ == '__main__':
    load_dotenv()
    unpack( roboflow_api_key=getenv( "ROBOFLOW_API_KEY" ), datasets_path = getenv( "DATASETS_PATH", './datasets' ) )
