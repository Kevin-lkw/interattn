CUDA_VISIBLE_DEVICES=7 python onlinetrain.py --config-name=setattn_mqar \
attn.type=vanilla \
wandb.log=true \
wandb.run_name=mqar_vanilla_dp1 \
out_dir=out-mqar-dp1/vanilla
