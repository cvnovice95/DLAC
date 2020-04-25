#Collect args.
while getopts :n:g:c: opts; do
    case $opts in
        n) num_nodes=$OPTARG ;;
        g) gpu_per_node=$OPTARG ;;
        c) cpu_per_node=$OPTARG ;;
        ?) ;;
    esac
done

# if num_nodes > 1: multi machine training
if [ ! $num_nodes ]; then
    num_nodes=1
fi

#export NCCL_DEBUG=INFO
## Don't touch these NCCL stuffs, it's maggic
#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=ib0
#export NCCL_TREE_THRESHOLD=0
#export NCCL_LL_THRESHOLD=0
#NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
#[ -z "$NCCL_IB_HCA" ] && NCCL_IB_HCA=mlx4_1;
#export NCCL_IB_HCA

# Begin tarining.
cd ..
for((rank=0;rank < $num_nodes;rank++));
do
    rlaunch --cpu=$cpu_per_node --gpu=$gpu_per_node --memory=$((1024*80)) --max-wait-time=1h \
    --charged-group=research_video --priority=Medium --rand-hostname -- \
    python3 \
        data_provider.py
    sleep 1
done