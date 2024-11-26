export CUDA_VISIBLE_DEVICES=0

python scripts/extract.py esm2_t33_650M_UR50D $1 $2 --repr_layers 33 --include mean --toks_per_batch 2048