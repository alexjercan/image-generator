# Simple Image Generation

### Quickstart

```console
docker build -t fastapi-sdxl .
docker run --device nvidia.com/gpu=all -p 8080:8000 -v $(pwd)/.hf_cache:/root/.cache/huggingface fastapi-sdxl:latest
```
