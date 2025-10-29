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
import wandb
def test_model(out_dir, cfg):
    dataset = cfg.data.dataset
    test_samples = cfg.data.test_samples
    test_length = cfg.data.test_length
    test_randomize = cfg.data.test_randomize
    
    device = cfg.system.device

    # 9) attn
    levelmax = cfg.attn.levelmax
    dataset = load_data(dataset, cfg.data, device=device)

    """
    test model part
    """
    log = True
    if log:
        exp = out_dir.split('/')[0]
        model = out_dir.split('/')[1]
        wandb.init(project=f"test-{exp}",name=model)

    print("Evaluating best model on test set...")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_model = ckpt['model']
    best_model_args = ckpt['model_args']
    X, Y = dataset.sample_batch('test',batch_size=test_samples, length = test_length, randomize = test_randomize) # fetch the very first batch
    if best_model is not None:
        best_model_args['levelrand'] = False  # turn off level randomization for testing
        for level in range(0, levelmax+1):
            best_model_args['level'] = level
            eval_model = GPT(GPTConfig(**best_model_args))
            eval_model.load_state_dict(best_model)
            eval_model.to(device)
            eval_model.eval()
            with torch.no_grad():
                logits, loss, acc = eval_model(X, Y, acc = True)
            print(f"level = {level}, test loss {loss.item():.4f}, test acc {acc.item():.3f}")
            if log:
                wandb.log({
                    "test_loss": loss.item(),
                    "test_acc": acc.item(),
                },step=level)
    if log: 
        wandb.finish()
@hydra.main(config_path="config", config_name="setattn_copy", version_base=None)
def main(cfg: DictConfig):
    for k in range(2):
        for v in range(2):
            for s in range(2):
                out_dir = f"out-copy/k{k}_v{v}_s{s}"
                test_model(out_dir, cfg)
    


if __name__ == "__main__":
    main()