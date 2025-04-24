from os import getenv, path
from dotenv import load_dotenv
from warnings import warn

from PIL import Image
from datetime import datetime
from typing import Dict, Optional, Any

from flask import Flask, Response, jsonify, request

import json
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from sklearn.metrics import precision_score, recall_score, f1_score

from model import CNN
from data_handler import Preproccessor, unpack, get_combined_dataset_dataloader

dry_run_datasets: bool = getenv( 'DRY_RUN_DATASETS' ) == 1

datetime_format: str = getenv( "DATETIME_FORMAT" ) or '%Y-%m-%dT%H:%M:%S'

datasets_path: str = getenv( "DATASETS_PATH" ) or './datasets'
surpress_dataset_warnings: bool = getenv( "DATASET_SURPRESS_WARNINGS", "0" ) == '1'
roboflow_api_key: Optional[ str ] = getenv( "ROBOFLOW_API_KEY" )
saved_model_path: str = getenv( "SAVED_MODEL_PATH", "" )

batch_size: int = int( getenv( "DATASET_BATCH_SIZE" ) or 32 )
shuffle: bool = getenv( "DATASET_SHUFFLE", "1" ) == '1'
num_workers: int = int( getenv( "DATASET_NUM_WORKERS" ) or 4 )

evaluate: bool = getenv( "EVALUATE" ) == '1'
evaluation_metrics_path: str = getenv( "EVALUATION_METRICS_PATH" ) or './model/model_evaluation_metrics.json'

train: bool = getenv( "TRAIN" ) == '1'
training_learning_rate: float = float( getenv( "TRAINING_LEARNING_RATE" ) or 0.001 )
training_epocs: int = int( getenv( "TRAINING_EPOCHS" ) or '10' )

webhook: Flask = Flask( getenv( "WEBHOOK_NAME" ) or 'Analyzer webhook' )
debug: bool = getenv( "DEBUG" ) == '1'
bind_address: str = getenv( "BIND_ADDRESS" ) or '0.0.0.0'
bind_port: int = int( getenv( "BIND_PORT" ) or 9000 )

model = CNN( device=torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' ) )

@webhook.route( "/analyze", methods=[ "POST" ] )
def analyze_image_webhook() -> Response:
    image_metadata: Dict[ str, str ] = request.get_json()
    image_path = image_metadata.get( 'uri', '' ).strip( 'file://' )

    # Analyze image and propogate errors
    assessment: bool = analyze_image( image_path )

    return jsonify( { 'assessment': assessment, 'assessment_timestamp': datetime.now().strftime( datetime_format ) } )

def analyze_image( image_path: str ) -> bool:
    image = Image.open( image_path )

    preprocess = Preproccessor()
    image_tensor = preprocess.process( image )
    image_tensor = image_tensor.unsqueeze( 0 )
    return model.test_image( image_tensor )

def validate_combined_dataset( surpress_warnings ) -> None:
    if path.exists( f'{ datasets_path }/combined_dataset' ):
        return

    warn( f'Combined dataset not present at path: "{ datasets_path }/combined_dataset"', UserWarning )
    warn( 'Please set "datasets_path" environment variable to point to the directory where the "combined_dataset" directory', UserWarning )

    if not surpress_warnings and input( 'Otherwise, auto unpack datasets to set directory? [y/N] ' ).lower() != 'y':
        exit( 0 )

    unpack( datasets_path, roboflow_api_key, dry_run_datasets )

def start_training() -> None:
    print( 'Validating Combined Dataset...' )
    validate_combined_dataset( surpress_warnings=surpress_dataset_warnings )

    print( 'Loading and preprocessing dataset...' )
    combined_dataset_dataloader = get_combined_dataset_dataloader( f'{ datasets_path }/combined_dataset/train', preprocess = True, batch_size=batch_size, shuffle = shuffle, num_workers = num_workers )

    print( 'Training Model...' )
    model.train_model( dataset=combined_dataset_dataloader, optimizer=Adam( model.parameters(), lr=training_learning_rate ), loss_fn=CrossEntropyLoss(), num_epochs=10 )

    print( 'Training Model Completed!' )
    model.save_model( saved_model_path )
    print( 'Model state dict saved to:', saved_model_path )

def start_evaluation() -> None:
    print( 'Validating Combined Dataset...' )
    validate_combined_dataset( surpress_warnings = surpress_dataset_warnings )

    print( 'Loading and preprocessing dataset...' )
    validation_loader = get_combined_dataset_dataloader( f'{ datasets_path }/combined_dataset/valid', preprocess = True, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers )

    print( 'Loading model...' )
    model.load_model( saved_model_path )

    print( 'Starting evaluation...' )
    accuracy: float = model.evaluate_model( validation_loader )

    all_preds, all_labels = model.get_predictions( validation_loader )
    precision: float = float( precision_score( all_labels, all_preds ) )
    recall: float = float( recall_score( all_labels, all_preds ) )
    f1: float = float( f1_score( all_labels, all_preds ) )
    timestamp: str = datetime.now().strftime( datetime_format )

    metrics: Dict[ str, Any ] = {
            'timestamp': timestamp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
    }

    with open( evaluation_metrics_path, 'w', encoding='utf-8' ) as file:
        json.dump( metrics, file, indent = 4 )

    checkpoint_path: str = evaluation_metrics_path.replace( '.json', f'_{timestamp}.json' )
    with open( checkpoint_path, 'w', encoding='utf-8' ) as file:
        json.dump( metrics, file, indent = 4 )

    print( f"Evaluation metrics saved to: {evaluation_metrics_path}" )
    print( f"Checkpointed metrics saved to: {checkpoint_path}" )



def start_analyzer() -> None:
    print( 'Loading Model...' )
    model.load_model( saved_model_path )

    print( 'Starting Webhook...' )
    webhook.run( host=bind_address, port=bind_port, debug=debug )

def main() -> None:
    print( 'Loading .env file if present...' )
    load_dotenv()

    if train:
        print( 'Starting Training...' )
        start_training()

    elif evaluate:
        print( 'Starting Evaluation...' )
        start_evaluation()

    else:
        print( 'Starting Analyzer...' )
        start_analyzer()

if __name__ == "__main__":
    main()
