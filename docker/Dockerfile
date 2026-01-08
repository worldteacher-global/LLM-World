FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.12 
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv git curl vim tmux && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Python3.12 as default "python"
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && \
    apt-get install -y gh
    
# Workspace
WORKDIR /app
COPY . .

# Install Python dependencies 
RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install jupyter
RUN pip install uv
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# Expose port
EXPOSE 8888

# Default command: Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

