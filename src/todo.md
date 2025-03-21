# TODO

## data_hanlder.py

- manual downloaded dataset as a provider
- automate as much preproccessing as possible
  - Image Size
  - Labels
  - Prune Low quality images
- apply image modifiers if needed
  - scale
  - rotation
  - color skew
  - focus/noise
- combine datasets
- split combined dataset into train/test
- save processed dataset
- be able to run seperately for preprocessing?

## webhook.py

- init flask webhook
- setup endpoint to analyze an image with image uri
- error handling for internal errors

## main.py

- default values for environment variables
  - WEBHOOK_NAME
  - TRAIN
  - DATASETS_PATH
  - COMPILED_MODEL_PATH
  - BIND_ADDRESS
  - BIND_PORT
- training proccess with a environment flag
  - if combined dataset, then start training
  - else, auto run unpacking, if not present and prompt user to preprocess
  - compile trained model/weights as well as export plots to images
  - propogate errors
- default running model with compiled model
  - use compiled model/weights
  - if no compiled model/weights, error and prompt user to train model first
  - on webhook, get image from uri and analyze
  - respond to request with assessment, assesment_timestamp, and image uri
  - propogate errors

## .env

- WEBHOOK_NAME: str
- TRAIN: bool
- DATASETS_PATH: str
- COMPILED_MODEL_PATH: str
- BIND_ADDRESS: str
- BIND_PORT: int
