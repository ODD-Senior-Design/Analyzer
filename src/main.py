from flask import Flask, Response, jsonify, request
from PIL import Image
from datetime import datetime
from typing import Dict
from os import getenv

from model import BinaryAlexNet
from data_handler import DataHandler, Preproccesser

datetime_format: str = getenv( "DATETIME_FORMAT" ) or '%Y-%m-%dT%H:%M:%S'

webhook: Flask = Flask( getenv( "WEBHOOK_NAME" ) or 'Analyzer webhook' )

@webhook.route( "/analyze", methods=[ "POST" ] )
def analyze_image_webhook() -> Response:
    image_metadata: Dict[ str, str ] = request.get_json()
    image_path = image_metadata.get( 'uri' )
    
    # Analyze image and propogate errors
    assessment: bool = analyze_image(  )
    
    return jsonify( { 'assessment': assessment, 'assessment_timestamp': datetime.now().strftime( datetime_format ) } )
    
    
def analyze_image(image_path: str, trained_model_save_path: str) -> bool:

    image = Image.open( image_path )
    
    model = BinaryAlexNet()
    model.load_model( trained_model_save_path )
    
    preprocess = Preproccesser()
    image_tensor = preprocess.proccess( image )
    image_tensor = image_tensor.unsqueeze(0)
    return model.test_image( image_tensor )
   