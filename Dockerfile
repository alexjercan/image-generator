FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app
COPY src/main.py .

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
