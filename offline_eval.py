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
import wandb
def load_model(out_dir):
    """
    load model part
    """
    ckpt = "ckpt_top1.pt"
    ckpt_path = os.path.join(out_dir, ckpt)
    ckpt = torch.load(ckpt_path,weights_only=False)
    cfg = ckpt['config']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device,dtype=ptdtype)
    train_corpus, validation_corpus, test_corpus_bins = load_data(config=cfg.data, num_bins=cfg.data.num_bins)
    voc = Voc()
    voc.create_vocab_dict(train_corpus)
    voc.noutputs = train_corpus.noutputs

    test_loader_bins = [Sampler(test_corpus_bin, voc, cfg.data.batch_size) for test_corpus_bin in test_corpus_bins]
    
    summary_acc = []
    for rank in range(1,6):
        ckpt = "ckpt_top{}.pt".format(rank)
        ckpt_path = os.path.join(out_dir, ckpt)
        ckpt = torch.load(ckpt_path,weights_only=False)
        cfg = ckpt['config']
        print("eval model with iter",ckpt['iter_num'], "val acc",ckpt['val_acc'])
        model = GPT(GPTConfig(**ckpt['model_args']))
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        for bin_idx, test_loader in enumerate(test_loader_bins):
            # perform testing

            test_loss = 0
            test_acc = 0
            test_iter_num = 0
            with torch.no_grad(), ctx:
                for j in range(0, len(test_loader.data), cfg.data.batch_size):
                    X_test, Y_test, _ = test_loader.get_batch(j)
                    X_test = X_test.to(device)
                    Y_test = Y_test.to(device)
                    _, loss, acc, _ = model(X_test, Y_test, acc = True, loss_type = cfg.data.loss_type)
                    test_loss += loss.item()
                    test_acc += acc.item()
                    test_iter_num += 1
                test_acc = test_acc / test_iter_num
                print(f"model{rank} Test Bin {bin_idx}: Avg Test Loss: {test_loss/test_iter_num:.4f}, Avg Test Acc: {test_acc:.4f}")
            summary_acc.append((rank, bin_idx, test_acc))
    for bin_idx in range(len(test_loader_bins)):
        accs = [acc for (rank, b_idx, acc) in summary_acc if b_idx == bin_idx]
        avg_acc = sum(accs) / len(accs)
        print(f"Average Test Acc for Bin {bin_idx}: {avg_acc:.4f}")
def main():
    out_dir='out-Boolean-3/vanilla_nope'
    load_model(out_dir)
    # test_model(out_dir)


if __name__ == "__main__":
    main()