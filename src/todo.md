# TODO

## data_hanlder.py

[ ] manual downloaded dataset as a provider

[ ] automate as much preproccessing as possible
  [ ] Image Size
  [ ] Labels
  [ ] Prune Low quality images

[ ] apply image modifiers if needed
  [ ] scale
  [ ] rotation
  [ ] color skew
  [ ] focus/noise

[ ] combine datasets
[ ] split combined dataset into train/test
[ ] save processed dataset
[ ] be able to run seperately for preprocessing?

## webhook.py

[ ] init flask webhook
[ ] setup endpoint to analyze an image with image uri
[ ] error handling for internal errors

## main.py

[x] default values for environment variables
  [x] WEBHOOK_NAME
  [x] DEBUG
  [x] DATASETS_PATH
  [x] COMPILED_MODEL_PATH
  [x] EVALUATION_METRICS_PATH
  [x] EVALUATE
  [x] TRAIN
  [x] BIND_ADDRESS
  [x] BIND_PORT
  [x] DATETIME_FORMAT

[x] training proccess with a environment flag
  [x] if combined dataset, then start training
  [x] else, auto run unpacking, if not present and prompt user to preprocess
  [ ] compile trained model/weights as well as export plots to images
  [ ] propogate errors

[ ] default running model with compiled model
  [ ] use compiled model/weights
  [ ] if no compiled model/weights, error and prompt user to train model first
  [ ] on webhook, get image from uri and analyze
  [ ] respond to request with assessment, assesment_timestamp, and image uri
  [ ] propogate errors

## .env

[ ] WEBHOOK_NAME: str
[ ] DEBUG: bool
[ ] TRAIN: bool (0/'' or 1)
[ ] EVALUATE: bool (0/'' or 1)
[ ] DATASETS_PATH: str
[ ] COMPILED_MODEL_PATH: str
[ ] EVALUATION_METRICS_PATH: str
[ ] ROBOFLOW_API_KEY: str
[ ] BIND_ADDRESS: str
[ ] BIND_PORT: int
[ ] DATETIME_FORMAT: str
