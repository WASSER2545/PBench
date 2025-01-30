import os
import yaml

from simulatedannealing import *
from replay_ta import *
from linearprogram_option import *
from random_send import *

if __name__ == '__main__': 
    directory = './configs/'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"processing {file}")
                    print(config)
                    ILP_work(config)
                    gen_ta(config)
                    replay_ta(config)

                    # gen_random(config)
                    # replay_random(config)