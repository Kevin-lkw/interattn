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
    "D_2": ("setattn_formal_Dn", {"data.dataset": "D_2", "data.num_par": 2}),
    "D_3": ("setattn_formal_Dn", {"data.dataset": "D_3", "data.num_par": 3}),
    "D_12": ("setattn_formal_Dn", {"data.dataset": "D_12", "data.num_par": 12}),
    "Parity": ("setattn_formal_Parity", {}),
    "AAStar": ("setattn_formal_AAStar", {}),
    "ABABStar": ("setattn_formal_ABABStar", {}),
    "Dyck-1": ("setattn_formal_Dyck", {}),
    "Shuffle-2": ("setattn_formal_Shuffle-2", {}),
    "Shuffle-4": ("setattn_formal_Shuffle-4", {}),
    "Counter-anbn": ("setattn_formal_Counter", {"data.dataset": "Counter-anbn", "data.num_par": 2, "optim.epochs": 5000}),
    "Counter-anbncn": ("setattn_formal_Counter", {"data.dataset": "Counter-anbncn", "data.num_par": 3, "optim.epochs": 5000}),
}
def run_single_experiment(task, attn_type, pos_enc, level, gpu):
    
    if task not in task_configs:
        print(f"❌ Unknown task: {task}")
        return False
    
    config_name, extra_params = task_configs[task]
    
    # 构建命令
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python offlinetrain.py",
        f"--config-name={config_name}",
        "wandb.log=true",
        f"wandb.project=setattn-formal-{task}-new",
        f"wandb.run_name=LG_{attn_type}" + f"_level{level}",
        f"out_dir=out-{task}/LG_{attn_type}" + f"_level{level}",
        f"attn.type={attn_type}",
        f"attn.level={level}",
        "attn.levelrand=False",
        f"attn.smaller_sets=False",
        f"model.pos_enc_type={pos_enc}" ,
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
    task, attn_type, pos_enc, level, gpu = args
    return run_single_experiment(task, attn_type, pos_enc, level, gpu)


def main():
    # 配置可用的GPU列表
    available_gpus = [3,3,3,4,4]   # 根据实际情况修改
    available_gpus = available_gpus 
    # 生成所有实验配置
    experiments = []
    attn_type = "setattn_linear"
    for task in task_configs.keys():
        if task not in ["D_2"]:
            continue
        for level in [0,1,2,3,4,5]:
            for pos_enc in ["nope"]:
                experiments.append((task, attn_type, pos_enc, level))
        
    # 为每个实验分配GPU（循环分配）
    gpu_cycle = cycle(available_gpus)
    experiments_with_gpu = [
        (task, attn_type, pos_enc, level, next(gpu_cycle))
        for task, attn_type, pos_enc, level in experiments
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
        for (task, attn_type, pos_enc, gpu), _ in failed_experiments:
            print(f"  - {task} | {attn_type}" + (f" | PE={pos_enc}" if pos_enc else "") + f" (GPU {gpu})")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(experiments_with_gpu)} experiments completed successfully!")


if __name__ == "__main__":
    main()
