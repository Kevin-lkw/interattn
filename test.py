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

@hydra.main(config_path="config", config_name="setattn_copy", version_base=None)
def main(cfg: DictConfig):
    # 1) I/O
    out_dir = cfg.out_dir

    # 3) data
    dataset = cfg.data.dataset
    test_samples = cfg.data.test_samples
    test_length = cfg.data.test_length
    test_randomize = cfg.data.test_randomize
    # 8) system
    device = cfg.system.device

    # 9) attn
    levelmax = cfg.attn.levelmax
    dataset = load_data(dataset, cfg.data, device=device)
    """
    test model part
    """
    print("Evaluating best model on test set...")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_model = ckpt['model']
    best_model_args = ckpt['model_args']
    if best_model is not None:
        best_model_args['levelrand'] = False  # turn off level randomization for testing
        for level in range(0, levelmax+1):
            best_model_args['level'] = level
            eval_model = GPT(GPTConfig(**best_model_args))
            eval_model.load_state_dict(best_model)
            eval_model.to(device)
            eval_model.eval()
            X, Y = dataset.sample_batch('test',batch_size=test_samples, length = test_length, randomize = test_randomize) # fetch the very first batch
            with torch.no_grad():
                logits, loss, acc = eval_model(X, Y, acc = True)
            print(f"level = {level}, test loss {loss.item():.4f}, test acc {acc.item():.3f}")



if __name__ == "__main__":
    main()