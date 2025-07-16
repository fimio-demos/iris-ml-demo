FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install any OS-level tools your app needs
RUN apt-get update && apt-get install -y \
      curl \
      git \
      gpg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code in
COPY . .

# Launch your test
CMD ["python", "cuda_test.py"]
