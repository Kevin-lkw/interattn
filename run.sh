 CUDA_VISIBLE_DEVICES=7 python offlinetrain.py \
 --config-name=setattn_formal_Shuffle-2 \
 attn.type=setattn_linear wandb.log=false wandb.project=SetAttn-formal-D_2-new wandb.run_name=LG_setattn_linear_level6_m6 out_dir=out-tmp attn.smaller_sets=False attn.level=6 attn.levelrand=false  model.pos_enc_type=nope