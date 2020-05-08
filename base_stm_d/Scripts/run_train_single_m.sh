# Begin tarining.
work_path=$(dirname $0)
python3 -u train_single_m.py \
  --config $1 \
2>&1|tee $work_path/full_log.txt