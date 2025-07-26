FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Remove the NVIDIA APT repos to avoid Package.gz errors
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
 && rm -f /etc/apt/sources.list.d/nvidia-ml.list \
 && apt-get update \
 && apt-get install -y \
      python3.10 \
      python3.10-distutils \
      python3-pip \
      curl \
      git \
      gpg \
 && rm -rf /var/lib/apt/lists/* \
 \
 # Register python3 and python aliases
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 \
 # Quick verification that 'python' now points to 3.10
 && python --version

# Copy your application code in
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Clear any inherited ENTRYPOINT so CMD runs directly
ENTRYPOINT []

# Unbuffered Python output
ENV PYTHONUNBUFFERED=1

# Use 'python' (aliased to python3.10) as our runtime
CMD ["python", "-u", "spiking_gpu_mem.py"]
