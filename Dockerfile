FROM python:3.12

RUN apt-get update && \
apt-get install -y git \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash/"]
