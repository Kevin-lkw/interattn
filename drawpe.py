import os
import torch
import matplotlib.pyplot as plt
import json
import math
def get_acc(dir):
    path = os.path.join(dir, 'acc.json')
    with open(path, 'r') as f:
        data = json.load(f)
        return data['avg_acc_per_bin'][0], data['avg_acc_per_bin'][1], data['std_acc_per_bin'][0], data['std_acc_per_bin'][1]
        # return data['summary_acc']['model1_bin0'], data['summary_acc']['model1_bin1'], 0, 0
def get_pt_acc(dir):
    path = os.path.join(dir, 'per_token_acc.json')
    with open(path, 'r') as f:
        data = json.load(f)
        return data['avg_acc_per_bin'][0], data['avg_acc_per_bin'][1], data['std_acc_per_bin'][0], data['std_acc_per_bin'][1]
        # return data['summary_acc']['model1_bin0'], data['summary_acc']['model1_bin1'], 0, 0
def draw(task):
    task_cfg_path = "./config/setattn_formal_{}.yaml".format(task)
    with open(task_cfg_path, 'r') as f:
        cfg = f.read()
    levelmax = math.floor(math.log2(int(cfg.split('upper_window: ')[1].split('\n')[0])))
    ind_acc, ood_acc, ind_std, ood_std = [], [], [], []
    pt_ind_acc, pt_ood_acc, pt_ind_std, pt_ood_std = [], [], [], []
    pes = ["nope","sinusoidal", "learned", "rope", "alibi", "t5"]
    for pe in pes:
        ind, ood, ind_s, ood_s = get_acc(f"out-{task}/vanilla_{pe}")
        pt_ind, pt_ood, pt_ind_s, pt_ood_s = get_pt_acc(f"out-{task}/vanilla_{pe}")
        ind_acc.append(ind)
        ood_acc.append(ood)
        ind_std.append(ind_s)
        ood_std.append(ood_s)
        pt_ind_acc.append(pt_ind)
        pt_ood_acc.append(pt_ood)
        pt_ind_std.append(pt_ind_s)
        pt_ood_std.append(ood_s)
    # plotting
    plt.figure()
    x = range(len(pes))
    width = 0.2  # 柱子的宽度
    plt.bar([i - width*0.5 for i in x], ind_acc, width,
            yerr=ind_std, capsize=3, label='IND')

    plt.bar([i + width*0.5 for i in x], ood_acc, width,
            yerr=ood_std, capsize=3, label='OOD')
    # plt.bar([i + width*0.5 for i in x], pt_ind_acc, width,
    #         yerr=pt_ind_std, capsize=3, label='PT-IND')
    # plt.bar([i + width*1.5 for i in x], pt_ood_acc, width,
    #         yerr=pt_ood_std, capsize=3, label='PT-OOD')
    plt.xlabel('PE')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{task}')
    plt.xticks(range(len(pes)), pes)
    plt.ylim(-0.1, 1.2)
    plt.grid(True)
    # plt.legend(fontsize='small', loc='lower right')
    plt.legend(
    fontsize=5,
    markerscale=0.6,
    # handlelength=1.5,
    # labelspacing=0.3,
    # handletextpad=0.4,
    # loc='upper left',
    # bbox_to_anchor=(1.01, 1.0),
    # borderaxespad=0.
    )
    # plt.tight_layout(rect=[0, 0, 0.99, 0.99])  
    os.makedirs(f'out-img', exist_ok=True)
    plt.savefig(f'out-img/{task}-acc_pe_avg5.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    for task in ["D_2","D_3","Parity","Shuffle-2","Shuffle-4","Boolean-3","Boolean-5"]:
        draw(task)
        print(f"Plot for task {task}")