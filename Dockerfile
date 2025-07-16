FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python 3.10, pip, and any OS-level tools
RUN apt-get update && apt-get install -y \
      python3.10 \
      python3.10-distutils \
      python3-pip \
      curl \
      git \
      gpg \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Python dependencies with pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run your CUDA test
CMD ["python3", "cuda_test.py"]
