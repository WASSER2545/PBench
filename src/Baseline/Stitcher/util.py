import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_random_state


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    check_random_state(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_data():
    """ Load data from a JSON file and preprocess it. """
    data_file = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/baseline_test/stitcher/profile.json"
    with open(data_file, "r") as f:
        data = json.load(f)

    benchmarks = set()
    for item in data:
        configurations = item["configuration"]
        for configuration in configurations:
            benchmarks.add((configuration["benchmark"], configuration["database"]))

    data_dict = defaultdict(list)
    for item in data:
        config_dict = {config['database']: {'terminal': config['terminal'], 'frequency': config['frequency']} for config in item['configuration']}
        
        for _, database in benchmarks:
            values = config_dict.get(database, {'terminal': 0, 'frequency': 0})
            # data_dict[database + '_terminal'].append(values['terminal'])
            # data_dict[database + '_frequency'].append(values['frequency'])
            # data_dict[database + '_auxiliary'].append(values['frequency'] * values['terminal'])
            
            # if values['frequency'] == 0:
            #     data_dict[database + '_auxiliarynew'].append(0)
            #     continue
            data_dict[database + '_auxiliarynew'].append(values['terminal'] * values['frequency'])
            
        data_dict["cputime"].append(item["cpu_time"])
        data_dict["scanbytes"].append(item["scan_bytes"] / (1024 ** 3))
    
    profile = pd.DataFrame(data_dict)
    return profile, benchmarks

def load_data_from_single_benchmark(benchmark_name):
    data_file = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/baseline_test/stitcher/train_data/{benchmark_name}.json"
    with open(data_file, "r") as f:
        data = json.load(f)

    benchmarks = set()
    for item in data:
        configurations = item["configuration"]
        for configuration in configurations:
            benchmarks.add((configuration["benchmark"], configuration["database"]))

    data_dict = defaultdict(list)
    for item in data:
        config_dict = {config['database']: {'terminal': config['terminal'], 'frequency': config['frequency']} for config in item['configuration']}
        
        for _, database in benchmarks:
            values = config_dict.get(database, {'terminal': 0, 'frequency': 0})
            data_dict[database + '_terminal'].append(values['terminal'])
            data_dict[database + '_frequency'].append(values['frequency'])
            # data_dict[database + '_auxiliary'].append(values['frequency'] * values['terminal'])
            
            # if values['frequency'] == 0:
            #     data_dict[database + '_auxiliarynew'].append(0)
            #     continue
            # data_dict[database + '_auxiliarynew'].append(values['terminal'] / values['frequency'])
            
        data_dict["cputime"].append(item["cpu_time"])
        data_dict["scanbytes"].append(item["scan_bytes"] / (1024 ** 3))
    
    profile = pd.DataFrame(data_dict)
    return profile, benchmarks