# Must use a Cuda version 11+
# Install linux packages

# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# copy from (https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

WORKDIR /

# Adding ENV Variables from BANANA
ARG SENTRY_DSN
ENV SENTRY_DSN=${SENTRY_DSN}

# Install git
RUN apt-get update && apt-get install -y git curl

# Upgrade pip
RUN pip3 install --upgrade pip

# Install poetry
RUN pip3 install poetry

# Generating requirements.txt
#   from pyproject.toml and poetry.lock (need both)
ADD pyproject.toml .
ADD poetry.lock .
RUN poetry export --without-hashes --format=requirements.txt > requirements.txt

# Install python packages from requirements.txt
#   (banana deps: sanic==22.6.2, accelerate)
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Download model weights
ADD download.py .
RUN python3 download.py


# Add your custom app code
#   app.py contains the init() and inference(post_body:dict) methods
ADD app.py .

EXPOSE 8000

# Alternatively... we could just use poetry itself LOL
# RUN poetry run python3 -u server.py
CMD python3 -u server.py
