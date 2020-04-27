work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --job-name=$3 \
python3 -u train.py \
2>&1|tee $work_path/full_log.txt
