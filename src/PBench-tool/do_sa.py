import os
import yaml

from simulatedannealing import *
from replay_sa import *
from linearprogram_option import *
from replay_interval import *
from random_send import *

if __name__ == '__main__': 
    directory = '/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/configs/1hexp/workload1h-5m-30s_4'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"processing {file}")
                    print(config)
                    ILP_work(config)
                    # replay_ILP(config)
                    # gen_sa(config)
                    # replay_sa(config)
                    # gen_random(config)
                    # replay_random(config)