FROM python:3.13-slim-bookworm

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./base_model/model_traced.pt ./base_model/model_traced.pt

COPY ./src ./src

ENV PYTHONUNBUFFERED=0
ENV TRAIN=0
ENV EVALUATE=0
ENV TEST=0
ENV SAVED_MODEL_PATH="./base_model/model_traced.pt"

CMD ["gunicorn", "-c", "gunicorn_config.py", "--logger-class=gunicorn_color.Logger", "--chdir=src", "app:app"]
