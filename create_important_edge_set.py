import time
import json
from proposed_attack_model import ProposedAttack
from utils import get_dataset_from_deeprobust

def get_important_edge_list_for_precompute(surrogate_model, dataset, defense_model):
    data_time = {}
    proposed_model = ProposedAttack(surrogate_model, dataset, defense_model)
    # start_time = time.time()
    important_edge_list = proposed_model.get_important_edge_list()
    # end_time = time.time()
    # running_time_seconds = end_time - start_time
    # running_time_minutes = running_time_seconds // 60
    # running_time_seconds %= 60
    # data_time["get_important_edge_set"] = f"running_time: {int(running_time_minutes)} minutes, {running_time_seconds} seconds"
    # with open(f"{dataset}_get_important_edge_set_running_time.json", "w") as json_file:
    #     json.dump(data_time, json_file)
    return important_edge_list

if __name__ == "__main__": 

    surrogate_model = 'gcn'
    # dataset = 'cora'
    defense_model = 'gcn'

    dataset_list = ['cora', 'citeseer', 'polblogs']
    # dataset_list = ['cora']
    important_edge_list_dict = {}
    for dataset in dataset_list:
        important_edge_list = get_important_edge_list_for_precompute(surrogate_model, dataset, defense_model)
        important_edge_list_dict[dataset] = important_edge_list
        important_edge_list_dict[dataset] = [list(item) for item in important_edge_list_dict[dataset]]

    # Save the dictionary to a JSON file
    # with open('./important_edge_list.json', 'w') as json_file:
    #     json.dump(important_edge_list_dict, json_file, indent=4)

    # dataset = 'ogbn-arxiv'
    # important_edge_list_dict = json.load(open('./important_edge_list.json', 'r', encoding='utf-8'))
    # important_edge_list = get_important_edge_list_for_precompute(surrogate_model, dataset, defense_model)
    # important_edge_list_dict[dataset] = important_edge_list
    # important_edge_list_dict[dataset] = [list(item) for item in important_edge_list_dict[dataset]]

    # with open('./important_edge_list.json', 'w') as json_file:
    #     json.dump(important_edge_list_dict, json_file, indent=4)

