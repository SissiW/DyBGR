# export CUDA_VISIBLE_DEVICES=0

# CUB dataset
# nohup python train.py -ds CUB -model vit_base_patch16_224_in21k -lam 0.3 -beta 0.5 -depth 2 -K 30 -T 3 -bs 405 -num_classes 100 -num_samples 9 -lr 3e-5 -lrt 8.5e-4 -max_epoch 50 -eval_ep [50] -resize 256 >/dev/null 1>>results  &

# Cars dataset
# nohup python train.py -ds Cars -model vit_base_patch16_224_in21k -alpha 0.07 -lam 0.1 -beta 0.5 -depth 2 -K 30 -T 3 -bs 405 -num_classes 98 -num_samples 9 -lr 3e-5 -lrt 1e-3 -max_epoch 150 -eval_ep [150] >/dev/null 1>>results &

# SOP dataset
# nohup python train.py -ds SOP -model vit_base_patch16_224_in21k -alpha 0.07 -lam 0.7 -beta 0.5 -depth 2 -K 30 -T 3 -bs 405 -num_classes 11318 -num_samples 9 -lr 3e-5 -lrt 1e-4 -max_epoch 30 -eval_ep [30] >/dev/null 1>>results &

# Inshop dataset
# nohup python train.py -ds Inshop -model vit_base_patch16_224_in21k -lam 0.3 -beta 0.5 -depth 2 -K 20 -T 5 -bs 405 -num_classes 3997 -num_samples 9 -lr 3e-5 -lrt 1e-4 -max_epoch 50 -eval_ep [50] >/dev/null 1>>results &


