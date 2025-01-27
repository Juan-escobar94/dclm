#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=dlcm_test
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem-per-cpu=8GB
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=p_scads_llm_secrets

set -x

module purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3
module load CMake/3.26.3

# activate environent here
DCLM_VENV_FOLDER="/data/cat/ws/juos515g-datacomp-juan/dclm-juan/.venv"

if [ -z "${DCLM_VENV_FOLDER}" ]; then
    echo "DCLM_VENV_FOLDER is not set or is empty"
    exit 1
else
    echo "DCLM_VENV_FOLDER set to: ${ENV_VAR}"
fi

if [ -z "${SLURM_GPUS_PER_TASK}" ]; then
    echo "SLURM_GPUS_PER_TASK is not set or is empty"
    SLURM_GPUS_PER_TASK=1
else
    echo "SLURM_GPUS_PER_TASK set to: ${SLURM_GPUS_PER_TASK}"
fi

source "$DCLM_VENV_FOLDER/bin/activate"

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
python3 ray_processing/process.py --source_ref_paths exp_data/datasets/raw_sources/dclm-pool-400m-1x_local.json --readable_name c4 --output_dir /data/horse/ws/jori152b-datacomp_data/ --config_path baselines/baselines_configs/c4.yaml --source_name cc_april_2019
# python -u simple-trainer.py "$SLURM_CPUS_PER_TASK"


