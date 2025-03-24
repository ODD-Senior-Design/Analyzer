from os import getenv, path
from dotenv import load_dotenv
from warnings import warn

from PIL import Image
from datetime import datetime
from typing import Dict, Optional, Any

from flask import Flask, Response, jsonify, request

from model import BinaryAlexNet
from data_handler import DataUnpacker, Preproccesser, DataLoader

datetime_format: str = getenv( "DATETIME_FORMAT" ) or '%Y-%m-%dT%H:%M:%S'

datasets_path: str = getenv( "DATASETS_PATH" ) or './datasets'
roboflow_api_key: Optional[ str ] = getenv( "ROBOFLOW_API_KEY" )
compiled_model_path: str = getenv( "COMPILED_MODEL_PATH" ) or './saved_models/model/model.pt'
evaluation_metrics_path: str = getenv( "EVALUATION_METRICS_PATH" ) or './model/model_evaluation_metrics.json'
evaluate: bool = getenv( "EVALUATE" ) == '1'
train: bool = getenv( "TRAIN" ) == '1'

webhook: Flask = Flask( getenv( "WEBHOOK_NAME" ) or 'Analyzer webhook' )
debug: bool = getenv( "DEBUG" ) == '1'
bind_address: str = getenv( "BIND_ADDRESS" ) or '0.0.0.0'
bind_port: int = int( getenv( "BIND_PORT" ) or 9000 )

@webhook.route( "/analyze", methods=[ "POST" ] )
def analyze_image_webhook() -> Response:
    image_metadata: Dict[ str, str ] = request.get_json()
    image_path = image_metadata.get( 'uri', '' ).replace( 'file:///', '/' )

    # Analyze image and propogate errors
    assessment: bool = analyze_image( image_path, compiled_model_path )

    return jsonify( { 'assessment': assessment, 'assessment_timestamp': datetime.now().strftime( datetime_format ) } )

def analyze_image( image_path: str, trained_model_save_path: str ) -> bool:
    image = Image.open( image_path )

    model = BinaryAlexNet()
    model.load_model( trained_model_save_path )

    preprocess = Preproccesser()
    image_tensor = preprocess.process( image )
    image_tensor = image_tensor.unsqueeze( 0 )
    return model.test_image( image_tensor )

# TODO: Implement function to train the model and save it to the compiled_model_path
def start_training() -> Any:
    pass

# TODO: Implement function to evaluate the model and save metrics to the evaluation_metrics_path
def start_evaluation() -> Any:
    pass

def validate_combined_dataset() -> None:
    if path.exists( f'{ datasets_path }/combined_datasets' ):
        return

    print( f'Combined dataset not present at path: "{ datasets_path }/combined_datasets"' )
    print( 'Please set "datasets_path" environment variable to point to the directory where the "combined_datasets" directory' )

    if input( 'Otherwise, auto unpack datasets to set directory? [y/N]' ).lower() != 'y':
        exit( 0 )

    if not roboflow_api_key:
        warn( 'Environment variable "ROBOFLOW_API_KEY" is not set. Keep in mind using Roboflow as a provider is not possible then.' )

    dataset_unpacker = DataUnpacker( datasets_save_path=datasets_path, roboflow_api_key=roboflow_api_key )
    dataset_unpacker.unpack_datasets()

    print( 'Datasets unpacked successfully! Please manually verify and combine datasets into a "combined_datasets" directory before training' )
    exit( 0 )

def main() -> None:
    print( 'Loading .env file if present...' )
    load_dotenv()

    if train:
        print( 'Starting training...' )
        validate_combined_dataset()
        start_training()

    elif evaluate:
        print( 'Starting evaluation...' )
        start_evaluation()

    else:
        print( 'Starting Analyzer...' )
        print( 'Starting Webhook...' )
        webhook.run( host=bind_address, port=bind_port, debug=debug )

if __name__ == "__main__":
    main()
