FROM python:3.13-slim-bookworm

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir ./base_model

RUN curl -L "https://nc.ranga-family.com/s/Dy9xnjYcPGP3YQK/download/model_traced.pt" -o ./base_model/model_traced.pt

COPY ./gunicorn_config.py ./gunicorn_config.py
COPY ./src ./src

ENV PYTHONUNBUFFERED=0
ENV TRAIN=0
ENV EVALUATE=0
ENV TEST=0
ENV SAVED_MODEL_PATH="./base_model/model_traced.pt"

CMD ["gunicorn", "-c", "gunicorn_config.py", "--logger-class=gunicorn_color.Logger", "--chdir=src", "main:app"]
