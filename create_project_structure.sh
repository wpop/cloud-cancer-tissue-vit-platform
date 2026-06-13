#!/usr/bin/env bash

set -e

echo "Creating project structure..."

# --------------------------------------------------
# Root files
# --------------------------------------------------

touch README.md
touch LICENSE
touch requirements.txt
touch pyproject.toml
touch .gitignore

# --------------------------------------------------
# Configs
# --------------------------------------------------

mkdir -p configs

touch configs/train.yaml
touch configs/inference.yaml
touch configs/aws.yaml

# --------------------------------------------------
# Data
# --------------------------------------------------

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/samples

# --------------------------------------------------
# Models
# --------------------------------------------------

mkdir -p models/checkpoints
mkdir -p models/exported

# --------------------------------------------------
# Outputs
# --------------------------------------------------

mkdir -p outputs/predictions
mkdir -p outputs/figures
mkdir -p outputs/logs

# --------------------------------------------------
# Notebooks
# --------------------------------------------------

mkdir -p notebooks

touch notebooks/01_dataset_exploration.ipynb
touch notebooks/02_train_vit.ipynb
touch notebooks/03_model_analysis.ipynb

# --------------------------------------------------
# Source code
# --------------------------------------------------

mkdir -p src

touch src/__init__.py
touch src/config.py

# --------------------------------------------------
# Data module
# --------------------------------------------------

mkdir -p src/data

touch src/data/__init__.py
touch src/data/dataset.py
touch src/data/transforms.py
touch src/data/datamodule.py

# --------------------------------------------------
# Models module
# --------------------------------------------------

mkdir -p src/models

touch src/models/__init__.py
touch src/models/vit_classifier.py
touch src/models/inference.py

# --------------------------------------------------
# Training module
# --------------------------------------------------

mkdir -p src/training

touch src/training/__init__.py
touch src/training/trainer.py
touch src/training/losses.py
touch src/training/metrics.py

# --------------------------------------------------
# API module
# --------------------------------------------------

mkdir -p src/api

touch src/api/__init__.py
touch src/api/app.py
touch src/api/routes.py
touch src/api/schemas.py

# --------------------------------------------------
# AWS module
# --------------------------------------------------

mkdir -p src/aws

touch src/aws/__init__.py
touch src/aws/s3_client.py
touch src/aws/upload.py

# --------------------------------------------------
# Visualization module
# --------------------------------------------------

mkdir -p src/visualization

touch src/visualization/__init__.py
touch src/visualization/probability_plot.py
touch src/visualization/attention_map.py

# --------------------------------------------------
# Utils module
# --------------------------------------------------

mkdir -p src/utils

touch src/utils/__init__.py
touch src/utils/logger.py
touch src/utils/helpers.py

# --------------------------------------------------
# Docker
# --------------------------------------------------

mkdir -p docker

touch docker/Dockerfile
touch docker/docker-compose.yml

# --------------------------------------------------
# Scripts
# --------------------------------------------------

mkdir -p scripts

touch scripts/train.sh
touch scripts/evaluate.sh
touch scripts/run_api.sh

chmod +x scripts/*.sh

# --------------------------------------------------
# Tests
# --------------------------------------------------

mkdir -p tests

touch tests/__init__.py
touch tests/test_api.py
touch tests/test_dataset.py
touch tests/test_model.py

echo
echo "Project structure successfully created!"
