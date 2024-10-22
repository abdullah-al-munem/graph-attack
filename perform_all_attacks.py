import os
import logging
import warnings
import pandas as pd
import ast
import argparse

from utils import get_dataset_from_deeprobust, get_target_node_list, get_miss_classification_original_dataset
from proposed_attack_model import start_attack_proposed_model
from state_of_the_art_attack_models import start_attack_RND, start_attack_FGA, start_attack_Nettack, start_attack_SGAttack, start_attack_IGAttack

warnings.simplefilter('ignore')

log_file_name = "./all_attack.log"

# Check if the file exists
if os.path.exists(log_file_name):
    if os.path.exists(f"{log_file_name}.bak"):
        os.remove(f"{log_file_name}.bak")
    os.rename(log_file_name, f"{log_file_name}.bak")
    # Remove the original file
    # os.remove(f"{log_file_name}")

# Configure logging to both console and file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Log to console
                        logging.FileHandler(log_file_name)  # Log to a file
                    ])

# Create a logger
logger = logging.getLogger(__name__)
logger.info('All attacks has been started...')


if __name__ == "__main__": 

    '''
    Todo List:

    1. cora-gcn (done.)
    2. cora-gin (done.)
    3. cora-rgcn  (done.)
    4. cora-graphsage (done.)

    5. citeseer-gcn (done.)
    6. citeseer-gin (done.)
    7. citeseer-rgcn
    8. citeseer-graphsage (done.)

    9. polblogs-gcn (done.)
    10. polblogs-gin (done.)
    11. polblogs-rgcn (done.)
    12. polblogs-graphsage (done)

    13. ogbn-gcn (running...)
    14. ogbn-gin
    15. ogbn-graphsage
    '''

    parser = argparse.ArgumentParser(description="Perform various attacks on GNN models")

    defense_model_list = ['gcn', 'gin', 'gat', 'graphsage', 'rgcn', 'mdgcn', 'jacgcn', 'svdgcn']
    dataset_list = ['cora', 'citeseer', 'polblogs', 'ogbn-arxiv', 'pubmed', 'BlogCatalog']

    # Set surrogate_model as optional with a default value
    parser.add_argument('--surrogate_model', type=str, default='gcn',
                        help='The surrogate model to use (default: gcn)')

    # Set dataset and defense_model as required arguments (no default values)
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset to use (required)')
    parser.add_argument('--defense_model', type=str, required=True,
                        help='The defense model to use (required)')
    
    # Parse the arguments
    args = parser.parse_args()

    # Assign arguments to variables
    surrogate_model = args.surrogate_model
    dataset = args.dataset
    defense_model = args.defense_model

    logger.info(f"Surrogate model: {surrogate_model}, Dataset: {dataset}, Defense model: {defense_model}")


    # surrogate_model = 'gcn'
    # dataset = 'cora'
    # defense_model = 'mdgcn'

    data = get_dataset_from_deeprobust(dataset=dataset)
    print("Dataset loaded...")
    budget_range = 7

    times = 5

    file_list = os.listdir('./')
    # print(csv_file_list)

    for time in range(1, times+1):
        # print(f"Running attacks for {time} time(s)......")
        # print(f"Proposed model started...")
        flg_proposed, flg_random, flg_fga, flg_nettack, flg_sgattack = 1, 1, 1, 1, 1
        csv_file_list = [file for file in file_list if ('.csv' in  file) and (str(time) in file)]

        for file in csv_file_list:
            _model = file[:-4].split('_')[0]
            _time = int(file[:-4].split('_')[-1])

            if _time == time and _model.lower() == 'proposed':
                flg_proposed = 0
            if _time == time and _model.lower() == 'random':
                flg_random = 0
            if _time == time and _model.lower() == 'fga':
                flg_fga = 0
            if _time == time and _model.lower() == 'nettack':
                flg_nettack = 0
            if _time == time and _model.lower() == 'sgattack':
                flg_sgattack = 0

        logger.info(f"Running attacks for {time} time(s)......")
        # "ugba, RGCN"
        if not (flg_proposed or flg_random or flg_fga or flg_nettack or flg_sgattack):
            continue

        if (not flg_proposed) or (not flg_random) or (not flg_fga) or (not flg_nettack):
            for file in csv_file_list:
                if str(time) in file:
                   df = pd.read_csv(file)
                   node_list = ast.literal_eval(df.loc[0,'node_list'])
                   break
        else:
            node_list = get_target_node_list(data)

        print(node_list)
        logger.info(f"Node list: {node_list}")
        # exit()
        # node_list = [929, 1163, 1347, 2077, 2082, 1049, 1185, 2260, 1669, 1486, 1104, 2429, 1207, 1068, 1891, 2379, 720, 2323, 2200, 1167, 1924, 1180, 413, 832, 2148, 159, 1878, 2481, 1448, 1745, 1505, 1547, 1938, 805, 354, 1321, 97, 811, 72, 366]
        
        # print("Targegt nodes are being selected...")
        logger.info(f"Targegt nodes are being selected...")

        # get_miss_classification_original_dataset(defense_model, dataset, node_list, time)

        if flg_proposed:
            logger.info(f"Proposed model attack has started...")
            start_attack_proposed_model(surrogate_model, dataset, defense_model, budget_range, node_list, time)

        if flg_random:
            logger.info(f"Random attack has started...")
            start_attack_RND(dataset, defense_model, budget_range, node_list, time)

        if flg_fga:
            logger.info(f"FGA has started...")
            start_attack_FGA(dataset, defense_model, budget_range, node_list, time)

        if flg_nettack:
            logger.info(f"Nettack has started...")
            start_attack_Nettack(dataset, defense_model, budget_range, node_list, time)
        
        if flg_sgattack:
            logger.info(f"SGAttack attack has started...")
            start_attack_SGAttack(dataset, defense_model, budget_range, node_list, time)


