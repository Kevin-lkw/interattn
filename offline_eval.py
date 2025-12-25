"""
Evaluation script for a model.
"""

import os
from contextlib import nullcontext

import torch
from model import GPTConfig, GPT
import hydra
from omegaconf import DictConfig
from data.formal_language.generate_data import load_data
from data.formal_language.dataloader import Sampler
from data.formal_language.utils.helper import Voc
import json
import math
import multiprocessing as mp
from multiprocessing import Pool
from itertools import cycle
def modify(x,y,bos):
    print("modify input")
    start = 0
    if bos == True:
        x[0][0] = 3
        y[0][0] = torch.tensor([-1,-1],dtype=y.dtype,device=y.device)
        start = 1
    x[0][start] = 1
    y[0][start] = torch.tensor([1,1],dtype=y.dtype,device=y.device)
    for i in range(start+1,x.size(1)):
        x[0][i] = 3 - x[0][i-1]
        y[0][i] = torch.tensor([1,1],dtype=y.dtype,device=y.device) if x[0][i]==1 else torch.tensor([1,0],dtype=y.dtype,device=y.device)
    # print("sample:", x[0]) 
def load_model(args):
    """
    load model part
    """
    out_dir, gpu, pt = args
    torch.cuda.set_device(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = 'cuda'
    ckpt = "ckpt_top1.pt"
    ckpt_path = os.path.join(out_dir, ckpt)
    ckpt = torch.load(ckpt_path,weights_only=False)
    cfg = ckpt['config']
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device,dtype=ptdtype)
    train_corpus, validation_corpus, test_corpus_bins = load_data(config=cfg.data, num_bins=cfg.data.num_bins)
    bos = getattr(cfg.data, 'bos', False)
    voc = Voc(bos)
    voc.create_vocab_dict(train_corpus)
    voc.noutputs = train_corpus.noutputs

    test_loader_bins = [Sampler(test_corpus_bin, voc, cfg.data.batch_size, bos=bos) for test_corpus_bin in test_corpus_bins]
    
    summary_acc = []
    for rank in range(1,6):
        ckpt = "ckpt_top{}.pt".format(rank)
        ckpt_path = os.path.join(out_dir, ckpt)
        ckpt = torch.load(ckpt_path,weights_only=False)
        cfg = ckpt['config']
        cfg.attn.levelmax = math.floor(math.log2(cfg.data.upper_window))
        print("eval model with iter",ckpt['iter_num'], "val acc",ckpt['val_acc'])
        model_args = ckpt['model_args']
        # import ipdb; ipdb.set_trace()
        model_args['attn'] = cfg.attn
        model = GPT(GPTConfig(**model_args))
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        for bin_idx, test_loader in enumerate(test_loader_bins):
            # perform testing

            test_loss = 0
            test_acc = 0
            pt_acc = 0
            test_iter_num = 0
            with torch.no_grad(), ctx:
                for j in range(0, len(test_loader.data), cfg.data.batch_size):
                    X_test, Y_test, _ = test_loader.get_batch(j)
                    X_test = X_test.clone()
                    Y_test = Y_test.clone()
                    # modify(X_test,Y_test,bos)
                    X_test = X_test.to(device)
                    Y_test = Y_test.to(device)
                    # import ipdb; ipdb.set_trace()
                    _, loss, acc, acc2 = model(X_test, Y_test, acc = True, loss_type = cfg.data.loss_type)
                    test_loss += loss.item()
                    test_acc += acc.item()
                    pt_acc += acc2.item()
                    test_iter_num += 1
                test_acc = test_acc / test_iter_num
                pt_acc = pt_acc / test_iter_num
                print(f"model{rank} Test Bin {bin_idx}: Avg Test Loss: {test_loss/test_iter_num:.4f}, Avg Test Acc: {test_acc:.4f}")
                print(f"model{rank} Test Bin {bin_idx}: Avg Per Token Acc: {pt_acc:.4f}")
            summary_acc.append((rank, bin_idx, test_acc if pt == False else pt_acc))
    avg_acc_list = []
    std_acc_list = []
    for bin_idx in range(len(test_loader_bins)):
        accs = [acc for (rank, b_idx, acc) in summary_acc if b_idx == bin_idx]
        avg_acc = sum(accs) / len(accs)
        variance = sum((x - avg_acc) ** 2 for x in accs) / len(accs) if accs else 0.0
        std_acc = math.sqrt(variance)
        avg_acc_list.append(avg_acc)
        std_acc_list.append(std_acc)
        print(f"Average Test Acc for Bin {bin_idx}: {avg_acc:.4f}")
    if pt == False:
        with open(os.path.join(out_dir, "acc.json"), "w") as f:
            sum_acc = {"model{}_bin{}".format(rank, bin_idx): acc for (rank, bin_idx, acc) in summary_acc}
            result = {"summary_acc": sum_acc, "avg_acc_per_bin": avg_acc_list, "std_acc_per_bin": std_acc_list}
            json.dump(result, f, indent=4)
    else :
        with open(os.path.join(out_dir, "per_token_acc.json"), "w") as f:
            sum_acc = {"model{}_bin{}".format(rank, bin_idx): acc for (rank, bin_idx, acc) in summary_acc}
            result = {"summary_acc": sum_acc, "avg_acc_per_bin": avg_acc_list, "std_acc_per_bin": std_acc_list}
            json.dump(result, f, indent=4)
def main():
    pt = False
    task = ["D_2","D_3","D_12","Parity","Shuffle-2","Shuffle-4","Boolean-3","Boolean-5","Tomita-3","Tomita-4","Tomita-5","Tomita-6","Tomita-7"]
    pes = ["nope", "sinusoidal", "learned", "rope", "alibi", "t5"]
    gpu_id = [0]
    gpu_cycle = cycle(gpu_id)
    task_list = []
    for t in ["Parity"]:
        # for level in range(0,9):
        #     for type in ["FX"]:
        #         name_str = f"setattn_linear_level{level}_{type}"
        #         out_dir=f"out-{t}/{name_str}"
        #         load_model(out_dir)
        # name_str = "vanilla_nope"
        # out_dir=f"out-{t}/{name_str}"
        # load_model(out_dir)
        # name_str = "linear_attention_nope"
        # out_dir=f"out-{t}/{name_str}"
        # load_model(out_dir)
        # name_str = "mamba"
        # out_dir=f"out-{t}/{name_str}"
        # load_model(out_dir)
        # name_str = "delta_net"
        # out_dir=f"out-{t}/{name_str}"
        # load_model(out_dir)
        for pe in ["rope"]:
            for depth in [8]:
                name_str = f"vanilla/{pe}/d{depth}"
                out_dir=f"out-{t}/{name_str}"
                # task_list.append((out_dir, next(gpu_cycle)))
                load_model((out_dir, 0, pt))

    # with Pool(processes=len(gpu_id)) as pool:
    #     results = pool.map(load_model, [ (out_dir, gpu, False) for (out_dir, gpu) in task_list ])
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()