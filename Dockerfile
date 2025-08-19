FROM python:3.12

RUN apt-get update && \
apt-get install -y git \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install jupyter
RUN pip install --no-cache-dir -r requirements.txt

# if runninng jupyter notebook
# EXPOSE 8888

# Run an interactive shell by default
CMD ["/bin/bash"] 

## Run a Jupyter Notebook server
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]