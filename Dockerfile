FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Remove the NVIDIA APT repo to avoid Package.gz errors
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
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT []

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "spiking_gpu_mem.py"]
