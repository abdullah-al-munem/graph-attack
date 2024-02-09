import os
import logging
import warnings
import pandas as pd
import ast

from utils import get_dataset_from_deeprobust, get_target_node_list, get_miss_classification_original_dataset
from state_of_the_art_attack_models import start_attack_IGAttack, start_attack_SGAttack

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
    3. cora-gat
    4. cora-graphsage

    5. citeseer-gcn (done.)
    6. citeseer-gin
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
    defense_model = 'gin'

    data = get_dataset_from_deeprobust(dataset=dataset)
    print("Dataset loaded...")
    budget_range = 7

    times = 5

    file_list = os.listdir('./')
    # print(csv_file_list)

    ### cora-gcn
    # all_node_list = [
    #     [929, 1554, 1504, 1163, 1342, 2406, 2307, 1185, 1347, 2260, 2341, 448, 1800, 1725, 2057, 136, 1995, 530, 1559, 941, 838, 1079, 1860, 477, 6, 836, 479, 2345, 1074, 1143, 2060, 545, 336, 1230, 1546, 231, 1254, 1397, 496, 1476],
    #     [929, 1554, 1504, 1163, 2082, 2307, 1185, 2406, 1820, 1639, 746, 2057, 2467, 1817, 682, 1276, 485, 1620, 368, 1908, 2075, 2100, 454, 332, 1607, 989, 730, 2124, 868, 426, 248, 173, 2251, 1254, 1872, 1683, 2006, 2256, 1904, 1924],
    #     [929, 1554, 1504, 1342, 2406, 1347, 1163, 1185, 1822, 838, 1582, 1591, 498, 1206, 1599, 2467, 1368, 2372, 124, 594, 2442, 173, 1479, 84, 2366, 1664, 405, 1192, 1588, 559, 1796, 1758, 1126, 855, 366, 1999, 122, 928, 1490, 2013],
    #     [929, 1554, 1504, 1342, 1347, 1163, 2406, 1820, 1185, 1779, 2155, 512, 745, 742, 320, 1583, 1744, 205, 440, 134, 1062, 13, 1751, 1073, 901, 44, 1811, 642, 1815, 2060, 624, 1158, 2192, 1176, 1441, 2217, 517, 2455, 2057, 603],
    #     [929, 1554, 1504, 838, 1342, 1347, 1820, 1163, 1185, 2307, 111, 2016, 1306, 1760, 73, 167, 1502, 2199, 169, 1120, 994, 1027, 1035, 401, 2010, 1898, 950, 1287, 1739, 1884, 9, 1200, 693, 2119, 11, 1490, 1048, 327, 2319, 2064],
    # ]

    #### citeseer_gcn
    # all_node_list = [
    #     [862, 1750, 1333, 1218, 1778, 1388, 688, 1687, 1904, 1694, 1262, 110, 217, 1325, 1929, 1384, 1270, 573, 775, 703, 590, 1360, 1196, 1913, 387, 712, 434, 1323, 627, 907, 347, 1241, 1990, 2059, 1192, 436, 1597, 887, 1542, 429],
    #     [862, 1750, 1388, 1694, 1904, 817, 1687, 2055, 1137, 1038, 1606, 1448, 81, 647, 1007, 1008, 2029, 917, 1159, 1169, 1037, 857, 858, 441, 16, 1903, 1148, 830, 687, 729, 1534, 852, 1777, 1216, 1174, 677, 1481, 2041, 524, 1924],
    #     [862, 1750, 817, 1320, 1388, 1452, 1453, 2055, 1289, 1137, 1893, 2072, 516, 1925, 168, 526, 93, 1435, 147, 66, 362, 95, 1442, 709, 1694, 228, 146, 1444, 1314, 1385, 60, 605, 1898, 115, 144, 554, 355, 1717, 713, 1051],
    #     [1750, 862, 1706, 1388, 1687, 888, 649, 1904, 817, 1311, 784, 812, 1563, 1731, 1625, 2029, 329, 555, 1831, 93, 1582, 69, 1963, 1432, 636, 1217, 2039, 2006, 1226, 966, 508, 724, 1331, 462, 908, 374, 2012, 1099, 414, 1011],
    #     [862, 1750, 817, 1452, 1320, 1388, 354, 1289, 1554, 1453, 1125, 319, 1952, 2040, 365, 1897, 1848, 784, 873, 191, 370, 362, 2021, 1378, 1119, 631, 2076, 847, 1992, 1586, 731, 470, 1892, 867, 1697, 1220, 2057, 376, 1924, 1295],
    # ]

    # ### polblogs-gcn
    # all_node_list = [
    #     [671, 47, 282, 357, 496, 339, 83, 881, 118, 444, 844, 894, 641, 389, 301, 948, 720, 468, 615, 62, 180, 638, 328, 482, 261, 637, 774, 354, 576, 215, 1127, 216, 944, 1167, 105, 690, 742, 1071, 785, 704],
    #     [47, 496, 357, 282, 339, 444, 252, 118, 83, 383, 574, 509, 537, 531, 173, 323, 1019, 267, 1018, 207, 308, 1136, 249, 81, 1114, 944, 123, 130, 683, 848, 454, 746, 601, 983, 951, 609, 34, 896, 657, 844],
    #     [671, 47, 282, 357, 496, 881, 339, 921, 83, 444, 1019, 301, 62, 1018, 942, 720, 106, 844, 640, 641, 1100, 839, 697, 285, 634, 64, 690, 1006, 745, 1105, 1044, 530, 1028, 1184, 323, 928, 725, 313, 638, 103],
    #     [671, 47, 282, 496, 357, 339, 881, 83, 118, 921, 1054, 62, 888, 1158, 942, 106, 2, 389, 1019, 49, 169, 731, 787, 723, 616, 927, 166, 502, 644, 773, 149, 305, 549, 86, 147, 612, 332, 343, 112, 1202],
    #     [671, 47, 496, 357, 282, 339, 83, 118, 881, 444, 341, 544, 641, 948, 62, 301, 640, 833, 709, 615, 910, 19, 575, 32, 753, 702, 1184, 1104, 142, 992, 667, 824, 882, 319, 1002, 870, 397, 966, 1069, 63],
    # ]

    ### cora-gin

    all_node_list = [
        [1554, 929, 1163, 1342, 1347, 2102, 2307, 2260, 2082, 1340, 2154, 1585, 1690, 1173, 1290, 1677, 677, 321, 1853, 2135, 1594, 2167, 1378, 1990, 2469, 2429, 1448, 538, 391, 2443, 1523, 319, 1741, 1944, 214, 996, 1217, 745, 1215, 1828],
        [1554, 929, 1342, 1347, 2102, 1163, 1541, 1340, 2082, 2007, 1318, 124, 441, 422, 640, 2020, 2155, 2380, 1839, 1978, 455, 529, 917, 2435, 1278, 2283, 304, 94, 1907, 182, 1718, 1116, 1272, 1173, 372, 1563, 406, 655, 926, 1938], 
        [1554, 929, 1342, 1163, 1347, 2007, 2102, 2082, 683, 2346, 321, 444, 2146, 1659, 975, 1207, 1220, 1979, 2299, 1658, 162, 877, 2438, 1627, 2244, 2256, 1045, 131, 2000, 1074, 526, 1025, 1320, 2348, 1727, 1156, 881, 1956, 1546, 455],
        [1554, 1504, 2082, 2406, 1342, 1049, 1347, 2007, 1669, 2077, 993, 2140, 602, 2085, 881, 896, 935, 2420, 172, 1315, 131, 2414, 1695, 2323, 1948, 517, 402, 2292, 547, 1080, 356, 34, 944, 757, 1781, 183, 2016, 2136, 2418, 189],
        [1554, 2007, 2082, 1049, 2077, 1021, 1823, 1347, 683, 2406, 1095, 1306, 764, 2340, 219, 2323, 746, 39, 2299, 488, 2361, 2302, 618, 1661, 473, 1478, 2286, 540, 708, 777, 2298, 2398, 1898, 1358, 2130, 173, 1985, 1194, 1778, 2229],
    ]

    for time in range(1, times+1):
        # print(f"Running attacks for {time} time(s)......")
        # print(f"Proposed model started...")
        flg_sgattack, flg_igattack, flg_fga = 1, 1, 1
        csv_file_list = [file for file in file_list if ('.csv' in  file) and (str(time) in file)]

        for file in csv_file_list:
            _model = file[:-4].split('_')[0]
            _time = int(file[:-4].split('_')[-1])

            if _time == time and _model.lower() == 'sgattack':
                flg_sgattack = 0
            # if _time == time and _model.lower() == 'igattack':
            #     flg_igattack = 0
            # if _time == time and _model.lower() == 'fga':
            #     flg_fga = 0

        logger.info(f"Running attacks for {time} time(s)......")

        # if not (flg_sgattack or flg_igattack or flg_fga):
        #     continue
        if not (flg_sgattack):
            continue

        node_list = all_node_list[time-1]

        print(node_list)
        logger.info(f"Node list: {node_list}")
        # exit()
        # node_list = [929, 1163, 1347, 2077, 2082, 1049, 1185, 2260, 1669, 1486, 1104, 2429, 1207, 1068, 1891, 2379, 720, 2323, 2200, 1167, 1924, 1180, 413, 832, 2148, 159, 1878, 2481, 1448, 1745, 1505, 1547, 1938, 805, 354, 1321, 97, 811, 72, 366]
        
        # print("Targegt nodes are being selected...")
        logger.info(f"Targegt nodes are being selected...")

        # get_miss_classification_original_dataset(defense_model, dataset, node_list, time)

        if flg_sgattack:
            logger.info(f"SGAttack attack has started...")
            start_attack_SGAttack(dataset, defense_model, budget_range, node_list, time)

        # if flg_igattack:
        #     logger.info(f"IGAttack has started...")
        #     start_attack_IGAttack(dataset, defense_model, budget_range, node_list, time)

        # if flg_nettack:
        #     logger.info(f"Nettack has started...")
        #     start_attack_Nettack(dataset, defense_model, budget_range, node_list, time)


