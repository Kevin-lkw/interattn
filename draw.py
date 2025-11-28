import os
import torch
import matplotlib.pyplot as plt

def main():
    LG_ind_acc, LG_ood_acc = [], []
    SM_ind_acc, SM_ood_acc = [], []
    task = "Shuffle-2"
    for level in range(0, 6):
        # LG
        out_dir_level = f"out-{task}/LG_setattn_linear_level{level}"
        ckpt_path = os.path.join(out_dir_level, 'bestloss.pt')
        ckpt = torch.load(ckpt_path, weights_only=False)
        lg_ind, lg_ood = ckpt['val_acc'][0], ckpt['val_acc'][1]
        LG_ind_acc.append(lg_ind)
        LG_ood_acc.append(lg_ood)

        # SM
        out_dir_level = f"out-{task}/SM_setattn_linear_level{level}"
        ckpt_path = os.path.join(out_dir_level, 'bestloss.pt')
        ckpt = torch.load(ckpt_path, weights_only=False)
        sm_ind, sm_ood = ckpt['val_acc'][0], ckpt['val_acc'][1]
        SM_ind_acc.append(sm_ind)
        SM_ood_acc.append(sm_ood)
    # import ipdb; ipdb.set_trace()
    # plotting
    levels = list(range(0, 6))
    plt.figure()

    # 颜色：C0 用于 IND，C1 用于 OOD
    # 线型：实线 = LG，虚线 = SM
    plt.plot(levels, LG_ind_acc, label='LG IND', linestyle='-',  marker='o', color='C0')
    plt.plot(levels, LG_ood_acc, label='LG OOD', linestyle='--',  marker='s', color='C0')    
    plt.plot(levels, SM_ind_acc, label='SM IND', linestyle='-', marker='o', color='C1')
    plt.plot(levels, SM_ood_acc, label='SM OOD', linestyle='--', marker='s', color='C1')
    
    plt.xlabel('Level')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{task}')
    plt.xticks(levels)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    os.makedirs('out-img', exist_ok=True)
    plt.savefig('out-img/model_accuracy.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()