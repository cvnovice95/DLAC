
cd ..
for((rank=0;rank < 1;rank++));
do
    python3 \
        evaluate_offline.py \
        --model_path='/data/ar_output/pretrain_model_zoo/tsn_resnet50_Seg5_RGB_HMDB51.pth' \
        --output_path=''
    sleep 1
done