import os
import torch
import matplotlib.pyplot as plt
import json
import math
def get_acc(dir):
    # if not os.path.exists(dir):
    #     raise ValueError(f"Directory {dir} does not exist.")
    # ckpt_path = os.path.join(dir, 'bestloss.pt')
    # ckpt = torch.load(ckpt_path, weights_only=False)
    # return ckpt['val_acc'][0], ckpt['val_acc'][1]
    path = os.path.join(dir, 'acc.json')
    with open(path, 'r') as f:
        data = json.load(f)
        return data['avg_acc_per_bin'][0], data['avg_acc_per_bin'][1], data['std_acc_per_bin'][0], data['std_acc_per_bin'][1]

def draw(task):
    task_cfg_path = "./config/setattn_formal_{}.yaml".format(task)
    with open(task_cfg_path, 'r') as f:
        cfg = f.read()
    levelmax = math.floor(math.log2(int(cfg.split('upper_window: ')[1].split('\n')[0])))
    LG_ind_acc, LG_ood_acc, LG_ind_std, LG_ood_std = [], [], [], []
    SM_ind_acc, SM_ood_acc, SM_ind_std, SM_ood_std = [], [], [], []
    FX_ind_acc, FX_ood_acc, FX_ind_std, FX_ood_std = [], [], [], []
    LG_levels = list(range(0, levelmax+2))
    SM_levels = list(range(0, levelmax+2))
    FX_levels = list(range(0, levelmax+2))
    for level in LG_levels:
        # LG
        lg_ind, lg_ood, lg_ind_std, lg_ood_std = get_acc(f"out-{task}/setattn_linear_level{level}_LG")
        LG_ind_acc.append(lg_ind)
        LG_ood_acc.append(lg_ood)
        LG_ind_std.append(lg_ind_std)
        LG_ood_std.append(lg_ood_std)
    for level in SM_levels:
        # SM
        sm_ind, sm_ood, sm_ind_std, sm_ood_std = get_acc(f"out-{task}/setattn_linear_level{level}_SM")
        SM_ind_acc.append(sm_ind)
        SM_ood_acc.append(sm_ood)
        SM_ind_std.append(sm_ind_std)
        SM_ood_std.append(sm_ood_std)
    for level in FX_levels:
        # FX
        fx_ind, fx_ood, fx_ind_std, fx_ood_std = get_acc(f"out-{task}/setattn_linear_level{level}_FX")
        FX_ind_acc.append(fx_ind)
        FX_ood_acc.append(fx_ood) 
        FX_ind_std.append(fx_ind_std)
        FX_ood_std.append(fx_ood_std)
    # import ipdb; ipdb.set_trace()
    # plotting
    vanilla_ind, vanilla_ood, vanilla_ind_std, vanilla_ood_std = get_acc(f"out-{task}/vanilla_nope")
    linear_ind, linear_ood, linear_ind_std, linear_ood_std = get_acc(f"out-{task}/linear_attention_nope")
    mamba_ind, mamba_ood, mamba_ind_std, mamba_ood_std = get_acc(f"out-{task}/mamba")
    delta_ind, delta_ood, delta_ind_std, delta_ood_std = get_acc(f"out-{task}/delta_net")
    plt.figure()

    # 颜色：C0 用于 IND，C1 用于 OOD
    # 线型：实线 = LG，虚线 = SM
    plt.errorbar(LG_levels, LG_ind_acc, yerr=LG_ind_std, label='LG IND', linestyle='-',  marker='o', color='C0',capsize=3)
    plt.errorbar(LG_levels, LG_ood_acc, yerr=LG_ood_std, label='LG OOD', linestyle='--',  marker='s', color='C0',capsize=3)    
    plt.errorbar(SM_levels, SM_ind_acc, yerr=SM_ind_std, label='SM IND', linestyle='-', marker='o', color='C1',capsize=3)
    plt.errorbar(SM_levels, SM_ood_acc, yerr=SM_ood_std, label='SM OOD', linestyle='--', marker='s', color='C1',capsize=3)
    plt.errorbar(FX_levels, FX_ind_acc, yerr=FX_ind_std, label='FX IND', linestyle='-', marker='o', color='C2',capsize=3)
    plt.errorbar(FX_levels, FX_ood_acc, yerr=FX_ood_std, label='FX OOD', linestyle='--', marker='s', color='C2',capsize=3)
    plt.axvline(
        x=levelmax,
        color='gray',        
        linestyle='-',    # 或 '--'
        linewidth=2.5,    # 加粗
        alpha=0.8,
        zorder=0          # 放在所有曲线后面，避免挡住
    )
    eps = 0.1
    std_eps = 0.00
    notes = []
    if vanilla_ind > 0.99 and vanilla_ood > 0.99:
        notes.append("Vanilla = 1")
    else:
        plt.axhline(y=vanilla_ind, color='C3', linestyle='-', label='Vanilla IND')
        if vanilla_ind_std>std_eps:
            plt.axhspan(
                vanilla_ind - vanilla_ind_std,
                vanilla_ind + vanilla_ind_std,
                color='C3',
                alpha=0.15 
            )
        plt.axhline(y=vanilla_ood, color='C3', linestyle='--', label='Vanilla OOD')
        if vanilla_ood_std>std_eps:
            plt.axhspan(
                vanilla_ood - vanilla_ood_std,
                vanilla_ood + vanilla_ood_std,
                color='C3',
            alpha=0.15  
            )
    if linear_ind > 0.99 and linear_ood > 0.99: 
        notes.append("Linear Attention = 1")
    elif linear_ind < 0.01 and linear_ood <0.01:
        notes.append("Linear Attention = 0")
    else :
        plt.axhline(y=linear_ind, color='C4', linestyle='-', label='LinearAttn IND')
        if linear_ind_std>std_eps:
            plt.axhspan(
                linear_ind - linear_ind_std,
                linear_ind + linear_ind_std,
                color='C4',
                alpha=0.15 
            )
        plt.axhline(y=linear_ood, color='C4', linestyle='--', label='LinearAttn OOD')
        if linear_ood_std>std_eps:
            plt.axhspan(
                linear_ood - linear_ood_std,
                linear_ood + linear_ood_std,
                color='C4',
            alpha=0.15  
            )

    if mamba_ind > 0.99 and mamba_ood > 0.99:
        notes.append("Mamba = 1")
    elif mamba_ind < 0.01 and mamba_ood <0.01:
        notes.append("Mamba = 0")
    else:
        plt.axhline(y=mamba_ind, color='C5', linestyle='-', label='Mamba IND')
        if mamba_ind_std>std_eps:
            plt.axhspan(
                mamba_ind - mamba_ind_std,
                mamba_ind + mamba_ind_std,
                color='C5',
                alpha=0.15 
            )
        plt.axhline(y=mamba_ood, color='C5', linestyle='--', label='Mamba OOD')
        if mamba_ood_std>std_eps:
            plt.axhspan(
                mamba_ood - mamba_ood_std,
                mamba_ood + mamba_ood_std,
                color='C5',
            alpha=0.15  
            )

    if delta_ind > 0.99 and delta_ood > 0.99:
        notes.append("DeltaNet = 1")
    elif delta_ind < 0.01 and delta_ood <0.01:
        notes.append("DeltaNet = 0")
    else:
        plt.axhline(y=delta_ind, color='C6', linestyle='-', label='DeltaNet IND')
        if delta_ind_std>std_eps:
            plt.axhspan(
                delta_ind - delta_ind_std,
                delta_ind + delta_ind_std,
                color='C6',
                alpha=0.15 
            )
        plt.axhline(y=delta_ood, color='C6', linestyle='--', label='DeltaNet OOD')
        if delta_ood_std>std_eps:
            plt.axhspan(
                delta_ood - delta_ood_std,
                delta_ood + delta_ood_std,
                color='C6',
                alpha=0.15  
            )
    plt.xlabel('Level')
    plt.ylabel('Validation Accuracy')
    notes = '\n' + ', '.join(notes) if notes else ''
    plt.title(f'{task}'+notes)
    plt.xticks(LG_levels)
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
    os.makedirs(f'out-{task}', exist_ok=True)
    plt.savefig(f'out-{task}/model_accuracy.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    for task in ["D_2","D_3","Parity","Shuffle-2","Shuffle-4","Boolean-3","Boolean-5"]:
        draw(task)
        print(f"Plot for task {task}")