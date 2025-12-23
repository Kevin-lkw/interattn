import os
import torch
import matplotlib.pyplot as plt
import json
import math
def get_acc(dir):
    path = os.path.join(dir, 'acc.json')
    with open(path, 'r') as f:
        data = json.load(f)
        # print("dir",dir)
        # print("ind_acc: "data['avg_acc_per_bin'][0], "ood_acc:", data['avg_acc_per_bin'][1],
        #       "ind_std:", data['std_acc_per_bin'][0], "ood_std:", data['std_acc_per_bin'][1])
        return data['avg_acc_per_bin'][0], data['avg_acc_per_bin'][1], data['std_acc_per_bin'][0], data['std_acc_per_bin'][1]
        # return data['summary_acc']['model1_bin0'], data['summary_acc']['model1_bin1'], 0, 0
def get_pt_acc(dir):
    path = os.path.join(dir, 'per_token_acc.json')
    with open(path, 'r') as f:
        data = json.load(f)
        return data['avg_acc_per_bin'][0], data['avg_acc_per_bin'][1], data['std_acc_per_bin'][0], data['std_acc_per_bin'][1]
        # return data['summary_acc']['model1_bin0'], data['summary_acc']['model1_bin1'], 0, 0
def draw(task):
    ind_acc, ood_acc, ind_std, ood_std = [], [], [], []
    d8_ind_acc, d8_ood_acc, d8_ind_std, d8_ood_std = [], [], [], []
    pes = ["nope","sinusoidal", "learned", "rope", "alibi", "t5"]
    for pe in pes:
        ind, ood, ind_s, ood_s = get_acc(f"out-{task}/vanilla/{pe}/d8")
        # pt_ind, pt_ood, pt_ind_s, pt_ood_s = get_pt_acc(f"out-{task}/vanilla_{pe}")
        # d4_ind, d4_ood, d4_ind_s, d4_ood_s = get_acc(f"out-{task}/vanilla_{pe}_d4")
        d8_ind, d8_ood, d8_ind_s, d8_ood_s = get_acc(f"out-{task}/vanilla/{pe}/d8_BOS")
        ind_acc.append(ind)
        ood_acc.append(ood)
        ind_std.append(ind_s)
        ood_std.append(ood_s)
        d8_ind_acc.append(d8_ind)
        d8_ood_acc.append(d8_ood)
        d8_ind_std.append(d8_ind_s)
        d8_ood_std.append(d8_ood_s)
        
    # plotting
    plt.figure()
    x = range(len(pes))
    width = 0.2  # 柱子的宽度
    plt.bar([i - width*1.5 for i in x], ind_acc, width,
            yerr=ind_std, capsize=3, label='IND',color='C0')

    plt.bar([i + width*0.5 for i in x], ood_acc, width,
            yerr=ood_std, capsize=3, label='OOD',color='C1')
    plt.bar([i - width*0.5 for i in x], d8_ind_acc, width,
            yerr=d8_ind_std, capsize=3, label='d16-IND',color='C0', alpha=0.65)
    plt.bar([i + width*1.5 for i in x], d8_ood_acc, width,
            yerr=d8_ood_std, capsize=3, label='d16-OOD',color='C1', alpha=0.65)
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
    os.makedirs(f'out-img/bos_d8', exist_ok=True)
    plt.savefig(f'out-img/bos_d8/{task}-acc.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # for task in ["D_2","D_3","D_12","Parity","Shuffle-2","Shuffle-4","Boolean-3","Boolean-5","Tomita-3","Tomita-4","Tomita-5","Tomita-6","Tomita-7"]:
    for task in ["D_1"]:
        draw(task)
        print(f"Plot for task {task}")