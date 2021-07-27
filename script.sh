# train on scannet v1 (PlaneNet dataset)
#CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.launch --nproc_per_node=3 --master_port 295025 train_planeTR.py
#CUDA_VISIBLE_DEVICES=7 python train_planeTR.py

# evaluate on sacnnet v1 (PlaneNet dataset)
#CUDA_VISIBLE_DEVICES=7 python eval_planeTR.py