import os
import torch
import matplotlib.pyplot as plt
import json
def get_acc(dir):
    # if not os.path.exists(dir):
    #     raise ValueError(f"Directory {dir} does not exist.")
    # ckpt_path = os.path.join(dir, 'bestloss.pt')
    # ckpt = torch.load(ckpt_path, weights_only=False)
    # return ckpt['val_acc'][0], ckpt['val_acc'][1]
    path = os.path.join(dir, 'acc.json')
    with open(path, 'r') as f:
        data = json.load(f)
        return data['avg_acc_per_bin'][0], data['avg_acc_per_bin'][1]

def main():
    LG_ind_acc, LG_ood_acc = [], []
    SM_ind_acc, SM_ood_acc = [], []
    task = "Parity"
    LG_levels = list(range(0, 9))
    SM_levels = list(range(0, 9))
    for level in LG_levels:
        # LG
        lg_ind, lg_ood = get_acc(f"out-{task}/setattn_linear_level{level}_LG")
        LG_ind_acc.append(lg_ind)
        LG_ood_acc.append(lg_ood)
    for level in SM_levels:
        # SM
        sm_ind, sm_ood = get_acc(f"out-{task}/setattn_linear_level{level}_SM")
        SM_ind_acc.append(sm_ind)
        SM_ood_acc.append(sm_ood)
    # import ipdb; ipdb.set_trace()
    # plotting
    vanilla_ind, vanilla_ood = get_acc(f"out-{task}/vanilla_nope")
    print(f"Vanilla: IND acc = {vanilla_ind}, OOD acc = {vanilla_ood}")
    linear_ind, linear_ood = get_acc(f"out-{task}/linear_attention_nope")

    plt.figure()

    # 颜色：C0 用于 IND，C1 用于 OOD
    # 线型：实线 = LG，虚线 = SM
    plt.plot(LG_levels, LG_ind_acc, label='LG IND', linestyle='-',  marker='o', color='C0')
    plt.plot(LG_levels, LG_ood_acc, label='LG OOD', linestyle='--',  marker='s', color='C0')    
    plt.plot(SM_levels, SM_ind_acc, label='SM IND', linestyle='-', marker='o', color='C1')
    plt.plot(SM_levels, SM_ood_acc, label='SM OOD', linestyle='--', marker='s', color='C1')
    
    plt.axhline(y=vanilla_ind, color='C2', linestyle='-', label='Vanilla IND')
    plt.axhline(y=vanilla_ood, color='C2', linestyle='--', label='Vanilla OOD')
    if linear_ind < 0.99 or linear_ood <0.99: 
        plt.axhline(y=linear_ind, color='C3', linestyle='-', label='Linear Attention IND')
        plt.axhline(y=linear_ood, color='C3', linestyle='--', label='Linear Attention OOD')
    plt.xlabel('Level')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{task}')
    plt.xticks(LG_levels)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    os.makedirs('out-img', exist_ok=True)
    plt.savefig('out-img/model_accuracy.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()