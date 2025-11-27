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
def test_model(out_dir):
    """
    test model part
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log = False
    if log:
        exp = out_dir.split('/')[0]
        model = out_dir.split('/')[1]
        wandb.init(project=f"test-{exp}",name=model)

    print("Evaluating best model on test set...")
    ckpt_path = os.path.join(out_dir, 'bestloss.pt')
    ckpt = torch.load(ckpt_path,weights_only=False)
    cfg = ckpt['config']
    print("eval model with acc",ckpt['val_acc'])
    best_model = ckpt['model']
    best_model_args = ckpt['model_args']
    
    # dataloader
    train_corpus, valid_corpus_bins = load_data(config=cfg.data, num_bins=cfg.data.num_bins)
    voc = Voc()
    voc.create_vocab_dict(train_corpus)
    voc.noutputs = train_corpus.noutputs

    train_loader = Sampler(train_corpus, voc, cfg.data.batch_size)
    val_loader_bins = [Sampler(val_corpus_bin, voc, cfg.data.batch_size) for val_corpus_bin in valid_corpus_bins]
    
    
    with torch.no_grad():
        eval_model = GPT(GPTConfig(**best_model_args))
        eval_model.load_state_dict(best_model)
        eval_model.to(device)
        eval_model.eval()
        for bin,val_loader in enumerate(val_loader_bins):
            for j in range(0, len(val_loader.data), cfg.data.batch_size):
                X_val, Y_val, _ = val_loader.get_batch(j)
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                eval_model(X_val, Y_val, acc = True, loss_type = cfg.data.loss_type, visualize = True)
    if log: 
        wandb.finish()
def main():
    out_dir='out-setattn_formal_Dn/h2_setattn_linear_level0'
    test_model(out_dir)


if __name__ == "__main__":
    main()