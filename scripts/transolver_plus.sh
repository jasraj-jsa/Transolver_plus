export CUDA_VISIBLE_DEVICES=0
# random port
port=$(shuf -i 10000-65535 -n 1)

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=$port \
#     main_airplane.py \
#     --nb_epochs 200 \
#     --fold_id 0 \
#     --dataset airplane \
#     --cfd_model=transolver_plus \
#     --data_dir /aircraft_data/ \
#     --save_dir /aircraft_data/ \
#     --eval 1

python main_car.py \
  --data_dir ./dataset/mlcfd_data/car_preprocessed \
  --save_dir ./dataset/mlcfd_data/car_preprocessed \
  --preprocessed 1 \
  --fold_id 0 --gpu 0 --nb_epochs 200 --lr 0.001