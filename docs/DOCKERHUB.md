# Embed API

REST API for generating text embeddings using the BGE-M3 model.

## Features

- **Dense embeddings**: 1024-dimensional vectors for semantic search
- **Sparse embeddings**: Sparse vectors for hybrid search  
- **ColBERT embeddings**: Multi-vector representation for precise matching
- **GPU support**: NVIDIA CUDA support for acceleration
- **Automatic memory management**: Model with idle timeout
- **RESTful API**: Clean REST endpoints for embeddings and model management

## Available Distributions

### Docker Images

Available variants:

- `cpu-latest`: CPU-only version with smaller image size
- `gpu-latest`: GPU-enabled version with CUDA support
- `latest`: Points to GPU-enabled version (same as `gpu-latest`)

Tagged versions are also available:

- `cpu-v1.0.0`, `gpu-v1.0.0`, `v1.0.0` etc.

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/get-started/)

### Using Docker

- **CPU version:**

    ```bash
    docker run -d -p 8000:8000 \
      -e DEVICE=cpu \
      ggwozdz/embed-api:cpu-latest
    ```

- **GPU version:**

    ```bash
    docker run -d -p 8000:8000 \
      -e DEVICE=cuda \
      --gpus all \
      ggwozdz/embed-api:gpu-latest
    ```

### Using Docker Compose

- Create a `docker-compose.yml` file with the following content and run `docker-compose up`:

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

## API Features

### Generate Embeddings

- Request:

    ```bash
    curl -X POST "http://localhost:8000/embeddings" \
      -H "Content-Type: application/json" \
      -d '{
        "texts": ["Hello world"],
        "include_dense": true,
        "include_sparse": true,
        "include_colbert": true
      }'
    ```

- Response:

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
        }
      ]
    }
    ```

### Model Status

- Request:

    ```bash
    curl -X GET "http://localhost:8000/model/status"
    ```

- Response:

    ```json
    {
      "is_loaded": true,
      "model_name": "BAAI/bge-m3",
      "device": "cpu"
    }
    ```

### Health Check

- Request:

    ```bash
    curl -X GET "http://localhost:8000/healthcheck"
    ```

- Response:

    ```json
    {
      "status": "OK"
    }
    ```

## Configuration

The application uses environment variables for configuration. Below are the available options:

| Variable | Description | Default value |
|----------|-------------|---------------|
| `DEVICE` | Device: `cpu` or `cuda` | `cpu` |
| `LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
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

## Documentation

Full documentation available at: [GitHub Repository](https://github.com/ggwozdz90/embed-api)
