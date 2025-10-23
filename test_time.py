


import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"1"

import time
import json


import pandas as pd
import argparse
import torch
import gc
from utils import get_dataset_from_deeprobust, get_target_node_list, get_miss_classification_original_dataset
from time_experiment_proposed_attack_model import start_attack_proposed_model
from time_experiment_state_of_the_art_attack_models import start_attack_RND, start_attack_FGA, start_attack_Nettack, start_attack_SGAttack
def flush_gpu_memory():
    """Flush GPU memory and cache for consistent timing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        gc.collect()
        print("GPU memory and cache flushed")
    else:
        print("CUDA not available, skipping GPU memory flush")

def clear_system_cache():
    """Clear system caches"""
    gc.collect()
    # Force Python garbage collection
    for i in range(3):
        gc.collect()
    print("System cache cleared")

def convert_time(seconds):
    minutes = seconds // 60
    seconds %= 60
    return minutes, seconds
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run time experiments for graph neural network attacks')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['cora', 'citeseer', 'polblogs', 'blogcatalog'],
                       help='Dataset to use for experiments')
    parser.add_argument('--defense_model', type=str, required=True,
                       choices=['gcn', 'gin', 'gat', 'graphsage', 'rgcn', 'mdgcn', 'jacgcn', 'svdgcn'],
                       help='Defense model to use')
    parser.add_argument('--gpu_id', type=int, required=True,
                       help='GPU ID to use for computation')
    return parser.parse_args()

if __name__ == "__main__": 
    args = parse_arguments()
    flush_gpu_memory()
    clear_system_cache()

    # # Set GPU device
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    # print(f"Using GPU: {args.gpu_id}")


    '''
    Todo List:

    1. cora-gcn (done.)
    2. cora-gin (done.)
    3. cora-gat (done.)
    4. cora-graphsage (done.)

    5. citeseer-gcn (done.)
    6. citeseer-gin (ruunning....)
    7. citeseer-gat
    8. citeseer-graphsage

    9. polblogs-gcn (done.)
    10. polblogs-gin
    11. polblogs-gat
    12. polblogs-graphsage
    '''

    defense_model_list = ['gcn', 'gin', 'graphsage', 'rgcn', 'mdgcn', 'jacgcn', 'svdgcn']
    dataset_list = ['cora', 'citeseer', 'polblogs']
    dataset_list = ['blogcatalog']
    # python test_time.py --dataset polblogs --defense_model gcn --gpu_id 0
    # python test_time.py --dataset polblogs --defense_model gin --gpu_id 0
    # python test_time.py --dataset polblogs --defense_model graphsage --gpu_id 0
    # python test_time.py --dataset polblogs --defense_model rgcn --gpu_id 0
    # python test_time.py --dataset polblogs --defense_model jacgcn --gpu_id 0
    # python test_time.py --dataset polblogs --defense_model svdgcn --gpu_id 0

    surrogate_model = 'gcn'
    # dataset = 'cora'
    # defense_model = 'gcn'
    dataset = args.dataset
    defense_model = args.defense_model

    data = get_dataset_from_deeprobust(dataset=dataset)
    print("Dataset loaded...")
    total_budget_range = 7

    times = 5

    file_list = os.listdir('./')
    # print(csv_file_list)
    
    for time_ in range(1, times+1):
        print(f"\n=== Starting iteration {time_}/{times} ===")
        
        # Flush memory and cache before each iteration
        flush_gpu_memory()
        clear_system_cache()
        
        node_list = get_target_node_list(data)
        data_time = {}
        data_time.setdefault("Proposed_model", [])
        data_time.setdefault("Random_attack", [])
        data_time.setdefault("FGA", [])
        data_time.setdefault("Nettack", [])
        data_time.setdefault("SGAttack_attack", [])

        print(node_list)
        print(f"Node list: {node_list}")
        node_list = [node_list[0]]
        # print("Targegt nodes are being selected...")
        print(f"Targegt nodes are being selected...")
        
        for budget_range in range(1, total_budget_range+1):
            print(f"\nRunnig for budget {budget_range}")
            
            # Flush before each attack method
            flush_gpu_memory()
            clear_system_cache()
        
            print(f"Proposed model attack has started...")
            running_time_minutes, running_time_seconds = start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list, time_)
            data_time["Proposed_model"].append({
                "running_time": f"{int(running_time_minutes)} minutes, {running_time_seconds} seconds",
                "budget": budget_range
            })
            flush_gpu_memory()  # Flush after each attack

        
            print(f"Random attack has started...")
            running_time_minutes, running_time_seconds = start_attack_RND(dataset, defense_model, budget_range, node_list, time_)
            data_time["Random_attack"].append({
                "running_time": f"{int(running_time_minutes)} minutes, {running_time_seconds} seconds",
                "budget": budget_range
            })
            flush_gpu_memory()  # Flush after each attack

        
            print(f"FGA has started...")
            running_time_minutes, running_time_seconds = start_attack_FGA(dataset, defense_model, budget_range, node_list, time_)
            data_time["FGA"].append({
                "running_time": f"{int(running_time_minutes)} minutes, {running_time_seconds} seconds",
                "budget": budget_range
            })
            flush_gpu_memory()  # Flush after each attack

        
            print(f"Nettack has started...")
            running_time_minutes, running_time_seconds = start_attack_Nettack(dataset, defense_model, budget_range, node_list, time_)
            # data_time["Nettack"].append({
            #     "running_time": f"{int(running_time_minutes)} minutes, {running_time_seconds} seconds",
            #     "budget": budget_range
            # })
            flush_gpu_memory()  # Flush after each attack

            print(f"Nettack has started...")
            running_time_minutes, running_time_seconds = start_attack_Nettack(dataset, defense_model, budget_range, node_list, time_)
            data_time["Nettack"].append({
                "running_time": f"{int(running_time_minutes)} minutes, {running_time_seconds} seconds",
                "budget": budget_range
            })
            flush_gpu_memory()  # Flush after each attack
        
        
            print(f"SGAttack attack has started...")
            running_time_minutes, running_time_seconds = start_attack_SGAttack(dataset, defense_model, budget_range, node_list, time_)
            data_time["SGAttack_attack"].append({
                "running_time": f"{int(running_time_minutes)} minutes, {running_time_seconds} seconds",
                "budget": budget_range
            })
            flush_gpu_memory()  # Flush after each attack

            with open(f"running_time_{time_}_dataset_{dataset}_defense_{defense_model}.json", "w") as json_file:
                json.dump(data_time, json_file, indent=4, ensure_ascii=False)
        
        with open(f"running_time_{time_}_dataset_{dataset}_defense_{defense_model}.json", "w") as json_file:
            json.dump(data_time, json_file, indent=4, ensure_ascii=False)
