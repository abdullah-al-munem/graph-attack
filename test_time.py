import time
import json

import os
import logging
import warnings
import pandas as pd
import ast

from utils import get_dataset_from_deeprobust, get_target_node_list, get_miss_classification_original_dataset
from proposed_attack_model import start_attack_proposed_model
from state_of_the_art_attack_models import start_attack_RND, start_attack_FGA, start_attack_Nettack, start_attack_SGAttack, start_attack_IGAttack

def convert_time(seconds):
    minutes = seconds // 60
    seconds %= 60
    return minutes, seconds

if __name__ == "__main__": 

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

    defense_model_list = ['gcn', 'gin', 'gat', 'graphsage']
    dataset_list = ['cora', 'citeseer', 'polblogs']

    surrogate_model = 'gcn'
    dataset = 'cora'
    defense_model = 'gcn'

    data = get_dataset_from_deeprobust(dataset=dataset)
    print("Dataset loaded...")
    budget_range = 1

    times = 1

    file_list = os.listdir('./')
    # print(csv_file_list)
    data_time = {}
    for time_ in range(1, times+1):
        # print(f"Running attacks for {time} time(s)......")
        # print(f"Proposed model started...")
        flg_proposed, flg_random, flg_fga, flg_nettack, flg_sgattack = 1, 1, 1, 1, 1
        csv_file_list = [file for file in file_list if ('.csv' in  file) and (str(time) in file)]

        for file in csv_file_list:
            _model = file[:-4].split('_')[0]
            _time = int(file[:-4].split('_')[-1])

            if _time == time_ and _model.lower() == 'proposed':
                flg_proposed = 0
            if _time == time_ and _model.lower() == 'random':
                flg_random = 0
            if _time == time_ and _model.lower() == 'fga':
                flg_fga = 0
            if _time == time_ and _model.lower() == 'nettack':
                flg_nettack = 0
            if _time == time_ and _model.lower() == 'sgattack':
                flg_sgattack = 0

        print(f"Running attacks for {time_} time(s)......")

        if not (flg_proposed or flg_random or flg_fga or flg_nettack or flg_sgattack):
            continue

        if (not flg_proposed) or (not flg_random) or (not flg_fga) or (not flg_nettack):
            for file in csv_file_list:
                if str(time_) in file:
                   df = pd.read_csv(file)
                   node_list = ast.literal_eval(df.loc[0,'node_list'])
                   break
        else:
            node_list = get_target_node_list(data)

        print(node_list)
        print(f"Node list: {node_list}")
        # exit()
        # node_list = [929, 1163, 1347, 2077, 2082, 1049, 1185, 2260, 1669, 1486, 1104, 2429, 1207, 1068, 1891, 2379, 720, 2323, 2200, 1167, 1924, 1180, 413, 832, 2148, 159, 1878, 2481, 1448, 1745, 1505, 1547, 1938, 805, 354, 1321, 97, 811, 72, 366]
        node_list = [node_list[0]]
        # print("Targegt nodes are being selected...")
        print(f"Targegt nodes are being selected...")

        get_miss_classification_original_dataset(defense_model, dataset, node_list, time_)

        if flg_proposed:
            print(f"Proposed model attack has started...")
            start_time = time.time()
            start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list, time_)
            end_time = time.time()
            running_time_seconds = end_time - start_time
            running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
            data_time["Proposed_model"] = f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds"


        if flg_random:
            print(f"Random attack has started...")
            start_time = time.time()
            start_attack_RND(dataset, defense_model, budget_range, node_list, time_)
            end_time = time.time()
            running_time_seconds = end_time - start_time
            running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
            data_time["Random_attack"] = f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds"

        if flg_fga:
            print(f"FGA has started...")
            start_time = time.time()
            start_attack_FGA(dataset, defense_model, budget_range, node_list, time_)
            end_time = time.time()
            running_time_seconds = end_time - start_time
            running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
            data_time["FGA"] = f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds"

        if flg_nettack:
            print(f"Nettack has started...")
            start_time = time.time()
            start_attack_Nettack(dataset, defense_model, budget_range, node_list, time_)
            end_time = time.time()
            running_time_seconds = end_time - start_time
            running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
            data_time["Nettack"] = f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds"
        
        if flg_sgattack:
            print(f"SGAttack attack has started...")
            start_time = time.time()
            start_attack_SGAttack(dataset, defense_model, budget_range, node_list, time_)
            end_time = time.time()
            running_time_seconds = end_time - start_time
            running_time_minutes, running_time_seconds = convert_time(running_time_seconds)
            data_time["SGAttack_attack"] = f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds"
        
        with open("running_time.json", "w") as json_file:
            json.dump(data_time, json_file)


