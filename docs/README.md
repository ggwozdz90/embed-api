# Embed API

REST API for generating text embeddings using the BGE-M3 model.

## Features

- **Dense embeddings**: 1024-dimensional vectors for semantic search
- **Sparse embeddings**: Sparse vectors for hybrid search  
- **ColBERT embeddings**: Multi-vector representation for precise matching
- **GPU support**: NVIDIA CUDA support
- **Automatic memory management**: Model with idle timeout
- **Docker ready**: Ready-to-use Docker images with GPU support

## Installation and Setup

### Docker Images

The API is available in two variants:

- **CPU-only** (`ggwozdz/embed-api:cpu-latest`): Smaller image with CPU-only PyTorch
- **GPU-enabled** (`ggwozdz/embed-api:gpu-latest`): Includes CUDA support for GPU acceleration
- **Default** (`ggwozdz/embed-api:latest`): Points to the GPU-enabled version

### Docker (recommended)

```bash
# CPU version
docker run -d -p 8000:8000 \
  -e DEVICE=cpu \
  ggwozdz/embed-api:cpu-latest

# GPU version  
docker run -d -p 8000:8000 \
  -e DEVICE=cuda \
  --gpus all \
  ggwozdz/embed-api:gpu-latest

# Alternative: use unversioned tags (GPU version)
docker run -d -p 8000:8000 \
  -e DEVICE=cuda \
  --gpus all \
  ggwozdz/embed-api:latest
```

### Docker Compose

```yaml
services:
  # CPU version
  api-cpu:
    image: ggwozdz/embed-api:cpu-latest
    environment:
      - DEVICE=cpu
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"
  
  # GPU version
  api-gpu:
    image: ggwozdz/embed-api:gpu-latest
    environment:
      - DEVICE=cuda
      - LOG_LEVEL=INFO
    ports:
      - "8001:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Poetry (development)

```bash
git clone <repository>
cd embed-api
poetry install
poetry run python src/main.py
```

## API Endpoints

Full API documentation available at: `http://localhost:8000/docs`

### 1. Health Check

```http
GET /healthcheck
```

Checks the application status.

### 2. Generate Embeddings

```http
POST /embeddings
```

Generates embeddings for the provided texts.

**Request:**

```json
{
  "texts": ["Hello world", "Another text"],
  "include_dense": true,
  "include_sparse": true,
  "include_colbert": true
}
```

**Response:**

```json
{
  "embeddings": [
    {
      "text": "Hello world",
      "dense": [0.1, 0.2, 0.3, -0.1, 0.5, 0.8, -0.2, 0.4],
      "sparse": {
        "indices": [1, 5, 10, 15, 23, 45, 67],
        "values": [0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]
      },
      "colbert": [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3]
      ]
    },
    {
      "text": "Another text",
      "dense": [0.4, 0.5, 0.6, -0.3, 0.7, 0.2, -0.1, 0.9],
      "sparse": {
        "indices": [2, 7, 12, 18, 34, 56],
        "values": [0.9, 0.5, 0.2, 0.15, 0.08, 0.03]
      },
      "colbert": [
        [0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9],
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
      ]
    }
  ]
}
```

### 3. Model Status

```http
GET /model/status
```

Checks if the model is loaded.

**Response:**

```json
{
  "is_loaded": true,
  "model_name": "BAAI/bge-m3",
  "device": "cpu"
}
```

### 4. Model Management

```http
POST /model/load    # Load model
POST /model/unload  # Unload model
```

## Configuration

The application is configured via environment variables:

| Variable | Description | Default value |
|----------|-------------|---------------|
| `DEVICE` | Device: `cpu` or `cuda` | `cpu` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `FASTAPI_HOST` | Server host | `127.0.0.1` |
| `FASTAPI_PORT` | Server port | `8000` |
| `MODEL_IDLE_TIMEOUT` | Model timeout (seconds) | `60` |

## BGE-M3 Model

BGE-M3 is a multilingual embedding model offering:

- **Dense embeddings**: 1024-dimensional vectors for semantic search
- **Sparse embeddings**: Sparse vectors for hybrid search
- **ColBERT embeddings**: Multi-vector representation for precise matching
- **Multilingual support**: Support for multiple languages
- **Long texts**: Up to 8192 tokens

## License

MIT License
