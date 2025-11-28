 CUDA_VISIBLE_DEVICES=7 python offlinetrain.py \
 --config-name=setattn_formal_Dn \
 attn.type=setattn_linear wandb.log=false wandb.project=SetAttn-formal-tmp wandb.run_name=vanilla_rope attn.smaller_sets=True attn.level=2 attn.levelrand=false