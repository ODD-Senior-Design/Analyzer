from os import getenv, makedirs, path
from dotenv import load_dotenv
from warnings import warn

from PIL import Image
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

from flask import Flask, Response, jsonify, request

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

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
model_dropout: float = float( getenv( "MODEL_DROPOUT" ) or 0.5 )
training_weight_decay: float = float( getenv( "TRAINING_WEIGHT_DECAY" ) or 0 )
training_learning_rate: float = float( getenv( "TRAINING_LEARNING_RATE" ) or 0.001 )
training_epocs: int = int( getenv( "TRAINING_EPOCHS" ) or 10 )
evaluation_threshold: float = float( getenv( "EVALUATION_THRESHOLD" ) or 0.5 )

evaluate: bool = getenv( "EVALUATE" ) == '1'

test: bool = getenv( "TEST" ) == '1'

train: bool = getenv( "TRAIN" ) == '1'

webhook: Flask = Flask( getenv( "WEBHOOK_NAME" ) or 'Analyzer webhook' )
debug: bool = getenv( "DEBUG" ) == '1'
bind_address: str = getenv( "BIND_ADDRESS" ) or '0.0.0.0'
bind_port: int = int( getenv( "BIND_PORT" ) or 9000 )

model = CNN( model_kwargs={ 'dropout': model_dropout }, device=torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' ) )


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

    prediction = float( model.test_image( image_tensor )[1] )
    print( f"Image { image_path } has a {prediction * 100:.2f}% chance of Gingivitis; Verdict: { 'Positive' if prediction > 0.5 else 'Negative' }" )
    return prediction > 0.5

def validate_combined_dataset( surpress_warnings ) -> None:
    if path.exists( f'{ datasets_path }/combined_dataset' ):
        return

    warn( f'Combined dataset not present at path: "{ datasets_path }/combined_dataset"', UserWarning )
    warn( 'Please set "datasets_path" environment variable to point to the directory where the "combined_dataset" directory', UserWarning )

    if not surpress_warnings and input( 'Otherwise, auto unpack datasets to set directory? [y/N] ' ).lower() != 'y':
        exit( 0 )

    unpack( datasets_path, roboflow_api_key, dry_run_datasets )

def save_metrics( model_metrics: Tuple[ np.ndarray, np.ndarray ], raw_model_metrics: Tuple[ np.ndarray, np.ndarray ], metrics_save_dir: str, include_confusion_matrix: bool = False, testing = False ) -> Tuple[ str, Dict[ str, Any ], Optional[ str ], Optional[ pd.DataFrame ] ]:
    if metrics_save_dir == './metrics' and not path.exists( metrics_save_dir ):
        makedirs( metrics_save_dir, exist_ok = True )
    elif not path.exists( metrics_save_dir ):
        raise FileNotFoundError( f'Directory { metrics_save_dir } does not exist.' )

    metrics_save_dir = f'{ metrics_save_dir }/{ path.basename( saved_model_path ).split( '.' )[0] }_metrics'
    makedirs( metrics_save_dir, exist_ok=True )

    accuracy: float = float( accuracy_score( *model_metrics ) )
    precision: float = float( precision_score( *model_metrics ) )
    recall: float = float( recall_score( *model_metrics ) )
    f1: float = float( f1_score( *model_metrics ) )
    roc_auc: float = float( roc_auc_score( *raw_model_metrics ) )
    cr: Dict[ str, Any ] = classification_report(
        *model_metrics,
        target_names = [ "Healthy", "Gingivitis" ],
        output_dict = True
    )

    cm: np.ndarray = confusion_matrix( *model_metrics )
    timestamp: str = datetime.now().strftime( datetime_format )

    metrics: Dict[ str, Any ] = {
        'timestamp': timestamp,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    metrics_file_path: str = f'{ metrics_save_dir }/{ path.basename( saved_model_path ).split( "." )[0] }_{ "testing" if testing else "validation" }_metrics_{ timestamp }.json'
    confusion_matrix_file_path: Optional[ str ] = f'{ metrics_save_dir }/metrics_{ timestamp }.csv' if include_confusion_matrix else None

    confusion_matrix_df: Optional[ pd.DataFrame ] = pd.DataFrame(
        cm,
        index = [ "Actual Healthy", "Actual Gingivitis" ],
        columns = [ "Predicted Healthy", "Predicted Gingivitis" ]
    ) if include_confusion_matrix else None

    with open( metrics_file_path, 'w', encoding = 'utf-8' ) as f:
        json.dump( metrics | { 'classification_report': cr }, f, indent = 4 )

    if confusion_matrix_df is not None:
        confusion_matrix_df.to_csv( confusion_matrix_file_path, index = True )

    return metrics_file_path, metrics, confusion_matrix_file_path, confusion_matrix_df

def start_training() -> None:

    print( 'Validating Combined Dataset...' )
    validate_combined_dataset( surpress_warnings=surpress_dataset_warnings )

    print( 'Loading and preprocessing dataset...' )
    combined_dataset_dataloader = get_combined_dataset_dataloader( f'{ datasets_path }/combined_dataset/train', preprocess = True, batch_size=batch_size, shuffle = shuffle, num_workers = num_workers )

    model_dir = saved_model_path[ :saved_model_path.rfind( '/' ) ]
    epochs_metrics_path = f'{ model_dir }/{ path.basename( saved_model_path ).split( '.' )[0] }_metrics/{ path.basename( saved_model_path ).split( '.' )[0] }_training_epochs_metrics.csv'
    makedirs( f'{ model_dir }/{ path.basename( saved_model_path ).split( '.' )[0] }_metrics', exist_ok=True )

    print( 'Training Model...' )
    model.train_model( dataset=combined_dataset_dataloader, optimizer=Adam( model.parameters(), lr=training_learning_rate, weight_decay=training_weight_decay ), metrics_save_path=epochs_metrics_path , loss_fn = BCEWithLogitsLoss(), num_epochs=training_epocs )

    print( '\nTraining Model Completed!' )

    model_path = model.save_model( saved_model_path )
    print( 'Model state dict saved to:', model_path )

    model.plot_loss( save_path=f'{ model_dir }/{ path.basename( saved_model_path ).split( '.' )[0] }_metrics/{ path.basename( model_path ).split( '.' )[0] }_loss_chart_{ datetime.now().strftime( datetime_format ) }.png' )

def start_evaluation() -> None:

    print( 'Validating Combined Dataset...' )
    validate_combined_dataset( surpress_warnings = surpress_dataset_warnings )

    print( 'Loading and preprocessing dataset...' )
    validation_loader = get_combined_dataset_dataloader( f'{ datasets_path }/combined_dataset/valid', preprocess = True, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers )

    print( 'Loading model...' )
    model.load_model( saved_model_path )

    print( 'Temperature calibrating model...' )
    model.set_temperature( validation_loader )

    print( 'Saving calibrated model...' )
    model.save_model( saved_model_path )

    print( 'Starting evaluation...' )
    model.evaluate_model( validation_loader, evaluation_threshold )

    metrics_path = saved_model_path[ :saved_model_path.rfind( '/' ) ]
    raw_model_metrics = model.get_predictions( validation_loader, return_probs=True )
    model_metrics = model.get_predictions( validation_loader )
    evaluation_metrics_path, metrics, _, _ = save_metrics( model_metrics, raw_model_metrics, metrics_path, include_confusion_matrix=False, testing=False )

    print( classification_report( *model_metrics, target_names=[ "Gingivitis", "Healthy" ] ) )
    print( f'\nTest metrics:\n{ metrics }\n' )
    print( f"Evaluation metrics saved to: { evaluation_metrics_path }" )

def start_testing() -> None:

    print( 'Validating Combined Dataset...' )
    validate_combined_dataset( surpress_warnings = surpress_dataset_warnings )

    print( 'Loading and preprocessing dataset...' )
    testing_loader = get_combined_dataset_dataloader(
        f'{ datasets_path }/combined_dataset/test',
        preprocess = True,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers
    )

    print( 'Loading model...' )
    model.load_model( saved_model_path )

    print( 'Starting testing...' )
    model.evaluate_model( testing_loader, evaluation_threshold )

    metrics_path = saved_model_path[ :saved_model_path.rfind( '/' ) ]
    raw_model_metrics = model.get_predictions( testing_loader, return_probs=True )
    model_metrics = model.get_predictions( testing_loader )
    test_metrics_path, metrics, confusion_matrix_path, confusion_matrix_df = save_metrics( model_metrics, raw_model_metrics, metrics_path, include_confusion_matrix=True, testing=True )

    print( classification_report( *model_metrics, target_names=[ "Gingivitis", "Healthy" ] ) )
    print( f'\nTest metrics:\n{ metrics }\n' )
    print( f"Test metrics saved to: { test_metrics_path }" )
    print( f"Confusion Matrix:\n{ confusion_matrix_df }" )
    print( f"Confusion matrix saved to: { confusion_matrix_path }" )

    y_true_shuffled = np.random.permutation( model_metrics[ 0 ] )

    print("\n=== Shuffled Labels Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_true_shuffled, model_metrics[ 1 ]):.2%}")
    print(f"Precision: {precision_score(y_true_shuffled, model_metrics[ 1 ]):.2%}")
    print(f"Recall: {recall_score(y_true_shuffled, model_metrics[ 1 ]):.2%}")
    print(f"F1 Score: {f1_score(y_true_shuffled, model_metrics[ 1 ]):.2%}")

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

    elif test:
        print( 'Starting Testing...' )
        start_testing()

    else:
        print( 'Starting Analyzer...' )
        start_analyzer()

if __name__ == "__main__":
    main()
