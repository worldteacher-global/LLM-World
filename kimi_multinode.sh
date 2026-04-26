#!/bin/bash
#SBATCH --job-name=kimi-k2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:8
#SBATCH --partition=defq
#SBATCH --container-image=/home/marfok/vllm-image+latest.sqsh
#SBATCH --container-mounts=/home/marfok:/home/marfok
#SBATCH --no-container-entrypoint
#SBATCH --output=/home/marfok/LLM-World/kimi-%j.out

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
worker_node=${nodes_array[1]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node ($head_node_ip)"
echo "Worker node: $worker_node"

# Start Ray head on first node
srun --nodes=1 --ntasks=1 -w "$head_node" \
  --container-image=/home/marfok/vllm-image+latest.sqsh \
  --container-mounts=/home/marfok:/home/marfok \
  --no-container-entrypoint \
  /bin/bash -c "pip install ray 2>/dev/null && ray start --head --port=6379 --block" &

sleep 15

# Start Ray worker on second node
srun --nodes=1 --ntasks=1 -w "$worker_node" \
  --container-image=/home/marfok/vllm-image+latest.sqsh \
  --container-mounts=/home/marfok:/home/marfok \
  --no-container-entrypoint \
  /bin/bash -c "pip install ray 2>/dev/null && ray start --address=${head_node_ip}:6379 --block" &

sleep 15

# Start Jupyter on head node
srun --nodes=1 --ntasks=1 -w "$head_node" \
  --container-image=/home/marfok/vllm-image+latest.sqsh \
  --container-mounts=/home/marfok:/home/marfok \
  --no-container-entrypoint \
  /bin/bash -c "
    pip install ray 2>/dev/null
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    JUPYTER_PORT=\$(seq 40000 50000 | shuf | head -n 1)
    echo \"using port \$JUPYTER_PORT on \$(hostname)\"
    jupyter notebook --ip=0.0.0.0 --port=\$JUPYTER_PORT --no-browser --allow-root --notebook-dir=/home/marfok/LLM-World
  "

wait
