FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install any OS-level tools your app needs
RUN apt-get update && apt-get install -y \
      curl \
      git \
      gpg \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install -r requirements.txt

# Launch your test
CMD ["python", "cuda_test.py"]
