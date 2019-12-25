
cd ..
for((rank=0;rank < 1;rank++));
do
    python3 \
        data_provider.py
    sleep 1
done