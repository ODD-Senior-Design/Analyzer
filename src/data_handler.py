from roboflow import Roboflow
from typing import List, Dict, Union, Any
import json

class DataHandler():

    def __init__( self, datasets_save_path: str, roboflow_api_key: str = None, dataset_manifest: str = '../datasets/dataset_manifests.json' ):
        self.__rf = Roboflow( api_key=roboflow_api_key ) if roboflow_api_key else None
        self.__datasets_save_path = datasets_save_path
        self.__dataset_manifest = dataset_manifest

    def __deserialize_dataset_manifest( self ) -> List[ Dict[ str, Union[ str, int ] ] ]:
        with open( self.__dataset_manifest, 'r', encoding='utf-8' ) as f:
            return json.load( f )

    def load_datasets( self ) -> Dict[ str, Any ]:
        manifest = self.__deserialize_dataset_manifest()
        datasets = {}
        for dataset_metadata in manifest:

            match dataset_metadata.get( 'provider', '' ).lower():

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
