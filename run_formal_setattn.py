#!/usr/bin/env python3
"""
简化版批量实验脚本 - 用于快速测试和小规模实验
支持多GPU并行执行
"""

import subprocess
import sys
from multiprocessing import Pool
from itertools import cycle
task_configs = {
    "D_1": ("setattn_formal_D_1", {"data.dataset": "D_1", "data.num_par": 1, "optim.epochs": 1000}),
    "D_2": ("setattn_formal_Dn", {"data.dataset": "D_2", "data.num_par": 2}),
    "D_3": ("setattn_formal_Dn", {"data.dataset": "D_3", "data.num_par": 3}),
    "D_12": ("setattn_formal_Dn", {"data.dataset": "D_12", "data.num_par": 12}),
    "Parity": ("setattn_formal_Parity", {}),
    "AAStar": ("setattn_formal_AAStar", {}),
    "ABABStar": ("setattn_formal_ABABStar", {}),
    "Dyck-1": ("setattn_formal_Dyck-1", {}),
    "Dyck-2": ("setattn_formal_Dyck-2", {}),
    "Shuffle-2": ("setattn_formal_Shuffle-2", {}),
    "Shuffle-4": ("setattn_formal_Shuffle-4", {}),
    "Boolean-3": ("setattn_formal_Boolean-3", {}),
    # "Boolean-3-lg": ("setattn_formal_Boolean-3-lg", {}),
    "Boolean-5": ("setattn_formal_Boolean-5", {}),
    # "Counter-anbn": ("setattn_formal_Counter", {"data.dataset": "Counter-anbn", "data.num_par": 2, "optim.epochs": 5000}),
    # "Counter-anbncn": ("setattn_formal_Counter", {"data.dataset": "Counter-anbncn", "data.num_par": 3, "optim.epochs": 5000}),
    "Counter-2": ("setattn_formal_Counter-2", {}),
    "Counter-3": ("setattn_formal_Counter-3", {}),
    "Tomita-3": ("setattn_formal_Tomita-3", {}),
    "Tomita-4": ("setattn_formal_Tomita-4", {}),
    "Tomita-5": ("setattn_formal_Tomita-5", {}),
    "Tomita-6": ("setattn_formal_Tomita-6", {}),
    "Tomita-7": ("setattn_formal_Tomita-7", {}),
}
def run_single_experiment(task, attn_type, pos_enc, level, set_policy, depth, gpu):
    
    if task not in task_configs:
        print(f"❌ Unknown task: {task}")
        return False
    
    config_name, extra_params = task_configs[task]
    
    # 构建命令
    name_str = f"{attn_type}"
    if attn_type == "vanilla" or attn_type == "linear_attention":
        name_str += f"/{pos_enc}/d{depth}_shortcut_BOS_200epoch"
    elif attn_type == "setattn_linear":
        name_str += f"/level{level}" + "/" + ("SM" if set_policy == "small" else ("LG" if set_policy == "large" else "FX"))
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python offlinetrain.py",
        f"--config-name={config_name}",
        "wandb.log=false",
        f"wandb.project=setattn-formal-{task}",
        f"wandb.run_name={name_str}",
        f"out_dir=out-{task}/{name_str}",
        f"attn.type={attn_type}",
        f"attn.level={level}",
        "attn.levelrand=False",
        f"attn.set_policy={set_policy}",
        f"model.pos_enc_type={pos_enc}" ,
        f"model.n_layer={depth}",
        f"data.bos=True",
        f"optim.epochs=200"
    ]
    
    
    # 添加额外参数
    for key, value in extra_params.items():
        cmd.append(f"{key}={value}")
    
    command = " ".join(cmd)
    print(f"\n{'='*80}")
    print(f"Command: {command}")
    print(f"{'='*80}\n")
    
    # 执行
    result = subprocess.run(command, shell=True)
    
    if result.returncode == 0:
        print(f"✅ Completed: {task} | {attn_type}")
        return True
    else:
        print(f"❌ Failed: {task} | {attn_type}")
        return False


def run_experiment_wrapper(args):
    """Wrapper function for parallel execution"""
    task, attn_type, pos_enc, level, set_policy, depth, gpu = args
    return run_single_experiment(task, attn_type, pos_enc, level, set_policy, depth, gpu)

level_mapping = {
    "vanilla":[0],
    "linear_attention":[0],
    "mamba":[0],
    "delta_net":[0],
    "setattn_linear":[6,7],
}
smaller_mapping = {
    "vanilla":["small"],
    "linear_attention":["small"],
    "mamba":["small"],
    "delta_net":["small"],
    "setattn_linear":["small", "large", "fixed"],
}
pe_mapping = {
    "vanilla":["sinusoidal", "learned", "rope", "alibi","nope", "t5"],
    "linear_attention":["nope", "rope"],
    "mamba":["nope"],
    "delta_net":["nope"],
    "setattn_linear":["nope"],
}
depth_mapping = {
    "vanilla":[16],
    "linear_attention":[2],
    "mamba":[2],
    "delta_net":[2],
    "setattn_linear":[2],
}
def main():
    # 配置可用的GPU列表
    available_gpus = [1]   # 根据实际情况修改
    # 生成所有实验配置
    experiments = []
    for attn_type in ["vanilla"]:
        for task in ["Parity"]:
            for level in level_mapping[attn_type]:
                for pos_enc in ["nope"]:
                    for depth in depth_mapping[attn_type]:
                        for set_policy in ["fixed"]:
                            experiments.append((task, attn_type, pos_enc, level, set_policy, depth))
            
    # 为每个实验分配GPU（循环分配）
    gpu_cycle = cycle(available_gpus)
    experiments_with_gpu = [
        (task, attn_type, pos_enc, level, set_policy, depth, next(gpu_cycle))
        for task, attn_type, pos_enc, level, set_policy, depth in experiments
    ]
    
    print(f"\n{'='*80}")
    print(f"📊 Total experiments: {len(experiments_with_gpu)}")
    print(f"🎮 Using GPUs: {available_gpus}")
    print(f"🔄 Parallel workers: {len(available_gpus)}")
    print(f"{'='*80}\n")
    
    # 并行执行
    with Pool(processes=len(available_gpus)) as pool:
        results = pool.map(run_experiment_wrapper, experiments_with_gpu)
    
    # 检查结果
    failed_experiments = [
        (exp, result) for exp, result in zip(experiments_with_gpu, results) if not result
    ]
    
    if failed_experiments:
        print(f"\n❌ {len(failed_experiments)} experiment(s) failed:")
        for (task, attn_type, pos_enc, level, smaller, depth, gpu), _ in failed_experiments:
            print(f"  - {task} | {attn_type} | level={level}" + (f" | PE={pos_enc}" if pos_enc else "") + f" | smaller={smaller} depth={depth} (GPU {gpu})")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(experiments_with_gpu)} experiments completed successfully!")


if __name__ == "__main__":
    main()
