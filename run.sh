 CUDA_VISIBLE_DEVICES=7 python offlinetrain.py \
 --config-name=setattn_formal_Dn \
 attn.type=vanilla wandb.log=true wandb.project=SetAttn-formal-tmp wandb.run_name=vanilla_rope model.pos_enc_type=rope