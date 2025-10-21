"""
Evaluation script for a model.
"""

import os
from contextlib import nullcontext

import torch
from model import GPTConfig, GPT
import hydra
from omegaconf import DictConfig
from data.load import load_data

@hydra.main(config_path="config", config_name="setattn_mqar", version_base=None)
def main(cfg: DictConfig):
    # 1) I/O
    out_dir = cfg.out_dir
    init_from = cfg.init_from

    # 3) data
    dataset = cfg.data.dataset
    batch_size = cfg.data.samples


    # 8) system
    device = cfg.system.device
    dtype = cfg.system.dtype
    compile = cfg.system.compile

    # 9) attn
    attn = cfg.attn.type
    level = cfg.attn.level
    levelrand = cfg.attn.levelrand
    levelmax = cfg.attn.levelmax
    master_process = True

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    dataset = load_data(dataset, batch_size, cfg.data, device=device)

    # model init
    model_args = dict(attn=attn, level=level, levelrand=levelrand, levelmax=levelmax)
                    # start with model_args from command line
    assert init_from == 'resume'
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # test loop
    X, Y = dataset.sample_batch('test') # fetch the very first batch


if __name__ == "__main__":
    main()