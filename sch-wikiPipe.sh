#!/bin/bash
#SBATCH --job-name=dlcm_test
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --exclusive
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --account=p_scads_llm_secrets
#SBATCH --gres=gpu:4

set -x  # Enable debugging

# Set working directory
WKDIR_DCLM="/data/horse/ws/komo978d-datacomp/dclm"
cd "$WKDIR_DCLM"

module purge
module load release/24.04
module load GCCcore/12.3.0
module load Anaconda3/2023.09-0
module load CMake/3.26.3



if [ -z "${SLURM_GPUS_PER_TASK}" ]; then
    echo "SLURM_GPUS_PER_TASK is not set or is empty"
    SLURM_GPUS_PER_TASK=4
else
    echo "SLURM_GPUS_PER_TASK set to: ${SLURM_GPUS_PER_TASK}"
fi

# Debug: Check Python version before activation
which python
python --version

# Activate virtual environment
source "$WKDIR_DCLM/../dclm_env/bin/activate"

# Debug: Check Python version after activation
which python
python --version

echo "Checking GPU availability..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print('Torch sees CUDA:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count())"


# Ensure the correct PYTHONPATH
export PYTHONPATH="$WKDIR_DCLM/../dclm_env/lib/python3.9/site-packages:$PYTHONPATH"

# Debug: Ensure SLURM_GPUS_PER_TASK is valid
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"
if [ -z "$SLURM_GPUS_PER_TASK" ]; then
    export SLURM_GPUS_PER_TASK=4
fi

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done
# __doc_worker_ray_end__

# __doc_script_start__
# ray/doc/source/cluster/doc_code/simple-trainer.py
# 
python3 ray_processing/process.py --source_ref_paths exp_data/datasets/raw_sources/dclm-pool-400m-1x_local.json --readable_name dclm_wiki_as_reference --output_dir /data/horse/ws/jori152b-datacomp_data/ --config_path baselines/baselines_configs/wikiAsReference.yaml --source_name cc 
