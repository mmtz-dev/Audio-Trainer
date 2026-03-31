# Audio-Trainer

Train and fine-tune TTS voice models using GPT-SoVITS. Containerized with Docker for reproducible GPU-accelerated training and inference.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with 12GB+ VRAM (24GB recommended)
- 5-30 minutes of voice recordings per speaker

## Quick Start

```bash
# 1. Build Docker images
make build

# 2. Download pretrained models (first time only)
make setup

# 3. Place voice recordings in data/raw/
cp ~/my_recordings/*.wav data/raw/

# 4. Preprocess audio (normalize, denoise, slice, transcribe)
make preprocess

# 5. Train voice model
make train

# 6. Generate speech
make generate TEXT="Hello, this is my cloned voice."

# 7. Start API server + WebUI
make serve
```

## Interfaces

| Interface | Port | Description |
|-----------|------|-------------|
| FastAPI REST API | 8000 | Programmatic TTS: `POST /v1/tts`, `GET /v1/speakers` |
| GPT-SoVITS WebUI | 9874 | Browser UI for preprocessing, training, and inference |

## Project Structure

```
Audio-Trainer/
├── core/              # Config, constants
├── preprocessing/     # Audio validation, normalization, slicing, transcription
├── training/          # GPT-SoVITS training wrapper, configs, evaluation
├── inference/         # Synthesizer, FastAPI, batch generation, post-processing
├── cli/               # Typer CLI (preprocess, train, generate, serve)
├── docker/            # Dockerfiles and entrypoint scripts
├── configs/           # YAML configuration files
├── data/              # Audio data (gitignored)
├── checkpoints/       # Model checkpoints (gitignored)
└── outputs/           # Generated audio (gitignored)
```

## Configuration

All paths are configurable via (in priority order):
1. CLI flags (`--output-dir /path`)
2. Environment variables (`AUDIO_TRAINER_OUTPUT_DIR=/path`)
3. Config YAML files (`configs/*.yaml`)
4. Defaults

## Make Targets

```
make build          # Build Docker images
make setup          # Download pretrained models
make preprocess     # Preprocess raw audio
make train          # Train with default config
make train-quick    # Quick training (for testing)
make train-quality  # High-quality training
make generate       # Generate speech (TEXT="...")
make serve          # Start API + WebUI
make test           # Run tests
make clean          # Remove processed data
```

## API Usage

```bash
# Start server
make serve

# Generate speech
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "speaker": "my_voice"}' \
  -o output.wav

# List speakers
curl http://localhost:8000/v1/speakers
```

## Training Presets

| Preset | SoVITS Epochs | GPT Epochs | Use Case |
|--------|--------------|------------|----------|
| `quick` | 2 | 3 | Testing the pipeline |
| `default` | 8 | 15 | Standard training |
| `quality` | 20 | 30 | Best results |

## License

MIT
