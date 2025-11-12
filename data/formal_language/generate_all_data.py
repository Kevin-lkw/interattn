#!/usr/bin/env python3
"""
Script to generate all formal language datasets based on the configuration table.
"""

import subprocess
import sys
import os
from typing import Dict, List, Tuple

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Generating: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def generate_dataset(
    lang: str,
    dataset: str, 
    training_size: int,
    lower_window: int,
    upper_window: int,
    test_size: int,
    bins: int,
    len_incr: int,
    num_par: int = None,
    p_val: float = None,
    q_val: float = None,
    crl_n: int = None,
) -> bool:
    """Generate a single dataset with given parameters."""
    
    cmd = [
        "python", "data/formal_language/generate_data.py",
        "-lang", lang,
        "-dataset", dataset,
        "-training_size", str(training_size),
        "-lower_window", str(lower_window),
        "-upper_window", str(upper_window),
        "-test_size", str(test_size),
        "-bins", str(bins),
        "-len_incr", str(len_incr),
    ]
    
    # Add optional parameters
    if num_par is not None:
        cmd.extend(["-num_par", str(num_par)])
    if p_val is not None:
        cmd.extend(["-p_val", str(p_val)])
    if q_val is not None:
        cmd.extend(["-q_val", str(q_val)])
    if crl_n is not None:
        cmd.extend(["-crl_n", str(crl_n)])
    
    description = f"{dataset} (size={training_size}, range=[{lower_window},{upper_window}], bins={bins})"
    return run_command(cmd, description)

def main():
    """Generate all datasets according to the configuration table."""
    
    print("🚀 Starting generation of all formal language datasets...")
    
    # Track success/failure
    total_tasks = 0
    successful_tasks = 0
    failed_tasks = []
    
    # Configuration for all tasks
    tasks = [
        # # Context-Free Languages
        # # Counter Languages
        # ("Counter", "Counter-anbn", 50, 2, 100, 50, 3, 100, {"num_par": 2}),
        # ("Counter", "Counter-anbncn", 50, 3, 150, 50, 3, 150, {"num_par": 3}),
        # ("Counter", "Counter-anbncndn", 50, 4, 200, 50, 3, 200, {"num_par": 4}),
        
        # # Shuffle Languages
        # ("Shuffle", "Shuffle-2", 10000, 2, 50, 2000, 3, 50, {"num_par": 2, "p_val": 0.5, "q_val": 0.25}),
        # ("Shuffle", "Shuffle-4", 10000, 2, 100, 2000, 3, 50, {"num_par": 4, "p_val": 0.5, "q_val": 0.25}),
        # ("Shuffle", "Shuffle-6", 10000, 2, 100, 2000, 3, 50, {"num_par": 6, "p_val": 0.5, "q_val": 0.25}),
        
        # # Boolean Languages
        # ("Boolean", "Boolean-3", 10000, 2, 50, 2000, 3, 50, {"num_par": 3, "p_val": 0.5}),
        # ("Boolean", "Boolean-5", 10000, 2, 50, 2000, 3, 50, {"num_par": 5, "p_val": 0.5}),
        
        # # Dyck Languages
        # ("Dyck", "Dyck-1", 10000, 2, 50, 2000, 3, 50, {"num_par": 1, "p_val": 0.5, "q_val": 0.25}),
        
        # # Regular Languages
        # # Tomita Grammars
        # ("Tomita", "Tomita-1", 50, 2, 50, 100, 2, 50, {"num_par": 1}),
        # ("Tomita", "Tomita-2", 25, 2, 50, 50, 2, 50, {"num_par": 2}),
        # ("Tomita", "Tomita-3", 10000, 2, 50, 2000, 2, 50, {"num_par": 3}),
        # ("Tomita", "Tomita-4", 10000, 2, 50, 2000, 2, 50, {"num_par": 4}),
        # ("Tomita", "Tomita-5", 10000, 2, 50, 2000, 2, 50, {"num_par": 5}),
        # ("Tomita", "Tomita-6", 10000, 2, 50, 2000, 2, 50, {"num_par": 6}),
        # ("Tomita", "Tomita-7", 10000, 2, 50, 2000, 2, 50, {"num_par": 7}),
        
        # # Star-Free Languages  
        # ("CStarAnCStarBnCStar", "CStarAnCStarBnCStar", 10000, 5, 200, 1000, 2, 100, {"num_par": 5}),
        # ("CAB_n_ABD", "CAB_n_ABD", 10000, 1, 50, 2000, 2, 50, {}),
        # ("CStarAnCStar", "CStarAnCStar", 10000, 2, 50, 2000, 2, 50, {"num_par": 3}),
        ("D_n", "D2", 10000, 2, 100, 2000, 2, 100, {"num_par": 2}),
        ("D_n", "D3", 10000, 2, 100, 2000, 2, 100, {"num_par": 3}),
        ("D_n", "D4", 10000, 2, 100, 2000, 2, 100, {"num_par": 4}),
        ("D_n", "D12", 10000, 2, 100, 2000, 2, 100, {"num_par": 12}),
        
        # Parity
        ("Parity", "Parity", 10000, 2, 50, 2000, 2, 50, {}),
        
        # # Non-Star-Free Languages
        # ("AAStar", "AAStar", 250, 2, 500, 50, 2, 100, {"num_par": 2}),
        # ("AnStarA2", "AnStarA2", 125, 4, 500, 25, 2, 100, {"num_par": 4}),
        # ("ABABStar", "ABABStar", 125, 4, 500, 25, 2, 100, {"num_par": 4}),
    ]
    
    for task in tasks:
        total_tasks += 1
        lang, dataset, training_size, lower_window, upper_window, test_size, bins, len_incr, extra_params = task
        
        # Prepare parameters
        params = {
            'lang': lang,
            'dataset': dataset,
            'training_size': training_size,
            'lower_window': lower_window,
            'upper_window': upper_window,
            'test_size': test_size,
            'bins': bins,
            'len_incr': len_incr
        }
        
        # Add extra parameters
        params.update(extra_params)
        
        # Generate dataset
        success = generate_dataset(**params)
        
        if success:
            successful_tasks += 1
            print(f"✅ {dataset}: SUCCESS")
        else:
            failed_tasks.append(dataset)
            print(f"❌ {dataset}: FAILED")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print('='*60)
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"\n❌ Failed tasks:")
        for task in failed_tasks:
            print(f"  - {task}")
    else:
        print(f"\n🎉 All tasks completed successfully!")
    
    print(f"\nSuccess rate: {successful_tasks/total_tasks*100:.1f}%")

if __name__ == "__main__":
    
    main()