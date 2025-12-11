 CUDA_VISIBLE_DEVICES=7 python offlinetrain.py \
 --config-name=setattn_formal_Boolean-3 \
 attn.type=delta_net wandb.log=false wandb.project=SetAttn-formal-D_2-new wandb.run_name=LG_setattn_linear_level6_m6 out_dir=out-tmp attn.set_policy=fixed attn.level=0 attn.levelrand=false  model.pos_enc_type=nope