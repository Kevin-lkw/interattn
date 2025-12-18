"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import copy

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LinearLR

from model import GPTConfig, GPT
import hydra
from omegaconf import DictConfig, OmegaConf
from data.formal_language.generate_data import load_data
from data.formal_language.dataloader import Sampler
from data.formal_language.utils.helper import Voc


@hydra.main(config_path="config", config_name="setattn_mqar", version_base=None)
def main(cfg: DictConfig):
    # 1) I/O
    out_dir = cfg.out_dir
    eval_interval = cfg.eval_interval
    log_interval = cfg.log_interval
    eval_iters = cfg.eval_iters
    eval_only = cfg.eval_only
    always_save_checkpoint = cfg.always_save_checkpoint
    init_from = cfg.init_from

    # 2) wandb
    wandb_log = cfg.wandb.log
    wandb_project = cfg.wandb.project
    wandb_run_name = cfg.wandb.run_name

    # 3) data
    dataset_name = cfg.data.dataset
    batch_size = cfg.data.batch_size
    block_size = cfg.data.block_size
    gradient_accumulation_steps = cfg.data.gradient_accumulation_steps

    # 4) model
    n_layer = cfg.model.n_layer
    n_head = cfg.model.n_head
    n_embd = cfg.model.n_embd
    dropout = cfg.model.dropout
    bias = cfg.model.bias
    pos_enc_type = cfg.model.pos_enc_type

    # 5) optim
    learning_rate = cfg.optim.learning_rate
    max_iters = cfg.optim.max_iters
    weight_decay = cfg.optim.weight_decay
    beta1 = cfg.optim.beta1
    beta2 = cfg.optim.beta2
    grad_clip = cfg.optim.grad_clip

    # 6) lr_decay
    decay_lr = cfg.lr_decay.enabled
    warmup_iters = cfg.lr_decay.warmup_iters
    lr_decay_iters = cfg.lr_decay.lr_decay_iters
    min_lr = cfg.lr_decay.min_lr

    # 7) ddp
    backend = cfg.ddp.backend

    # 8) system
    device = cfg.system.device
    dtype = cfg.system.dtype
    compile = cfg.system.compile

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0 + seed_offset)
    torch.cuda.manual_seed(0 + seed_offset)
    np.random.seed(0 + seed_offset)

    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    train_corpus, validation_corpus, test_corpus_bins = load_data(config=cfg.data, num_bins=cfg.data.num_bins)
    voc = Voc(cfg.data.bos)
    voc.create_vocab_dict(train_corpus)
    voc.noutputs = train_corpus.noutputs
    train_loader = Sampler(train_corpus, voc, cfg.data.batch_size,cfg.data.bos)
    val_loader = Sampler(validation_corpus, voc, cfg.data.batch_size,cfg.data.bos)
    test_loader_bins = [Sampler(test_corpus_bin, voc, cfg.data.batch_size,cfg.data.bos) for test_corpus_bin in test_corpus_bins]
    
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0

    # attempt to derive vocab_size from the dataset
    meta_vocab_size = voc.nwords
    # model init
    # ensure level do not exceed log2 of upper_window
    cfg.attn.levelmax = math.floor(math.log2(cfg.data.upper_window))
    
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                        bias=bias, vocab_size=None, dropout=dropout, attn=cfg.attn, pos_enc_type=pos_enc_type)
                    # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        model_args['n_outputs'] = voc.noutputs
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
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
        iter_num = checkpoint['iter_num']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name = wandb_run_name, config=OmegaConf.to_container(cfg, resolve=True))
    sum_loss = 0
    sum_acc = 0
    top_k = 5
    top_ckpts = []
    break_flag = False
    for epoch in range(1,cfg.optim.epochs+1):
        print(f"Epoch {epoch}/{cfg.optim.epochs}")
        epoch_iter_num = 0 # number of iterations in the lifetime of this epoch
        for i in range(0, len(train_loader.data), batch_size):
            X, Y, _ = train_loader.get_batch(i)
            X = X.to(device)
            Y = Y.to(device)
            # import ipdb; ipdb.set_trace()
            t0 = time.time()
            raw_model = model.module if ddp else model # unwrap DDP container if needed

            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    _, loss, acc, _ = model(X, Y, acc = True, loss_type = cfg.data.loss_type)
                    sum_loss += loss.item()
                    sum_acc += acc.item()
                    epoch_iter_num += 1
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            

            if iter_num % log_interval == 0 and master_process and iter_num > 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                # lossf = loss.item() * gradient_accumulation_steps
                lossf = sum_loss / log_interval
                sum_loss = 0
                accf = sum_acc / log_interval
                sum_acc = 0
                dt = time.time() - t0
                t0 = time.time()
                print(f"iter {iter_num}: loss {lossf:.4f}, acc {accf:.3f}, dt {dt:.4f}")
                if wandb_log:
                    wandb.log({'iter': iter_num, 'loss': lossf, 'acc': accf, 'lr': lr}, step=iter_num)
            if iter_num % eval_interval == 0:
                # perform validation
                model.eval()
                sum_val_loss = 0
                with torch.no_grad(), ctx:
                    val_loss = 0
                    val_acc = 0
                    val_iter_num = 0
                    for j in range(0, len(val_loader.data), batch_size):
                        X_val, Y_val, _ = val_loader.get_batch(j)
                        X_val = X_val.to(device)
                        Y_val = Y_val.to(device)
                        _, loss, acc, _ = model(X_val, Y_val, acc = True, loss_type = cfg.data.loss_type)
                        val_loss += loss.item()
                        val_acc += acc.item()
                        val_iter_num += 1
                        sum_val_loss += loss.item()
                    val_acc = val_acc / val_iter_num
                    print(f" Avg Val Loss: {val_loss/val_iter_num:.4f}, Avg Val Acc: {val_acc:.4f}")

                    if wandb_log and master_process:
                        wandb.log({f'val/loss': val_loss/val_iter_num, f'val/acc': val_acc},step=iter_num)
                        
                    checkpoint = {
                        'model': copy.deepcopy(raw_model.state_dict()),
                        'optimizer': copy.deepcopy(optimizer.state_dict()),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'val_acc': val_acc,
                        'config': cfg,
                    }
                    top_ckpts.append((val_acc, checkpoint))
                    top_ckpts = sorted(top_ckpts, key=lambda x: x[0], reverse=True)[:top_k] 
                    
                    if len(top_ckpts) == top_k and top_ckpts[-1][0] >= 0.999:
                        print("Top-k Validation accuracy reached 99.9%, stopping training.")
                        break_flag = True
                        break
                model.train()
            iter_num += 1
        if break_flag:
            break
    if ddp:
        destroy_process_group()
    print("Training complete. Saving final checkpoints...")
    
    summary_acc = []
    for rank, (val_acc, checkpoint) in enumerate(top_ckpts):
        model_state = checkpoint['model']
        model.load_state_dict(model_state)
        model.eval()
        for bin_idx, test_loader in enumerate(test_loader_bins):
            # perform testing

            test_loss = 0
            test_acc = 0
            test_iter_num = 0
            with torch.no_grad(), ctx:
                for j in range(0, len(test_loader.data), batch_size):
                    X_test, Y_test, _ = test_loader.get_batch(j)
                    X_test = X_test.to(device)
                    Y_test = Y_test.to(device)
                    _, loss, acc, _ = model(X_test, Y_test, acc = True, loss_type = cfg.data.loss_type)
                    test_loss += loss.item()
                    test_acc += acc.item()
                    test_iter_num += 1
                test_acc = test_acc / test_iter_num
                print(f"model{rank+1} Test Bin {bin_idx}: Avg Test Loss: {test_loss/test_iter_num:.4f}, Avg Test Acc: {test_acc:.4f}")

                if wandb_log and master_process:
                    wandb.summary[f'model{rank+1}_test_bin{bin_idx}/acc'] = test_acc
            summary_acc.append((rank+1, bin_idx, test_acc))
        ckpt_path = os.path.join(out_dir, f'ckpt_top{rank+1}.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path} with val acc {val_acc:.4f}")
    # calc avg acc
    for bin_idx in range(len(test_loader_bins)):
        accs = [acc for r, b, acc in summary_acc if b == bin_idx]
        avg_acc = sum(accs) / len(accs)
        print(f"Average Test Acc for Bin {bin_idx}: {avg_acc:.4f}")
        if wandb_log and master_process:
            wandb.summary[f'test_bin{bin_idx}/acc'] = avg_acc
if __name__ == "__main__":
    main()