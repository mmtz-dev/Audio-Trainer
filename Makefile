.PHONY: setup preprocess train train-quick train-quality generate serve serve-api serve-webui test clean clean-all build help

DOCKER_COMPOSE = docker compose
TRAIN_CMD = $(DOCKER_COMPOSE) --profile training run --rm train
INFERENCE_CMD = $(DOCKER_COMPOSE) run --rm inference

# Default target
help:
	@echo "Audio-Trainer - Voice TTS Training Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make build          Build Docker images"
	@echo "  make setup          Download pretrained models (first-time)"
	@echo ""
	@echo "Training Pipeline:"
	@echo "  make preprocess     Preprocess raw audio in data/raw/"
	@echo "  make train          Train voice model (default config)"
	@echo "  make train-quick    Quick training (fewer epochs, for testing)"
	@echo "  make train-quality  High-quality training (more epochs)"
	@echo ""
	@echo "Inference:"
	@echo "  make generate TEXT=\"Hello world\"  Generate speech from text"
	@echo "  make serve          Start API (port 8000) + WebUI (port 9874)"
	@echo "  make serve-api      Start API server only (port 8000)"
	@echo "  make serve-webui    Start GPT-SoVITS WebUI only (port 9874)"
	@echo ""
	@echo "Utilities:"
	@echo "  make test           Run tests"
	@echo "  make clean          Remove processed data (keep raw)"
	@echo "  make clean-all      Remove all generated data and checkpoints"

# Build Docker images
build:
	$(DOCKER_COMPOSE) --profile training build

# First-time setup
setup:
	$(TRAIN_CMD) setup

# Preprocessing
preprocess:
	$(TRAIN_CMD) python -m cli.main preprocess

# Training
train:
	$(TRAIN_CMD) python -m cli.main train --preset default

train-quick:
	$(TRAIN_CMD) python -m cli.main train --preset quick

train-quality:
	$(TRAIN_CMD) python -m cli.main train --preset quality

# Inference
generate:
	$(INFERENCE_CMD) python -m cli.main generate --text "$(TEXT)"

# Serving
serve:
	$(DOCKER_COMPOSE) up inference

serve-api:
	$(DOCKER_COMPOSE) run --rm -p 8000:8000 inference api

serve-webui:
	$(DOCKER_COMPOSE) run --rm -p 9874:9874 inference webui

# Testing
test:
	$(TRAIN_CMD) python -m pytest tests/ -v

# Cleanup
clean:
	rm -rf data/processed/*
	rm -rf outputs/*
	rm -rf logs/*

clean-all: clean
	rm -rf checkpoints/finetuned/*
	rm -rf data/raw/*
