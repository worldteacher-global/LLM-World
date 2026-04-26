FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y git build-essential python3 python3-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install uv
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install vllm
RUN pip install jupyter
RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8888
EXPOSE 7860

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
