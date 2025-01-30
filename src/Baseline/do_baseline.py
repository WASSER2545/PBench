import os
import yaml

from Stitcher.stitcher import *
from Stitcher.replay import *

from CAB.gen_redset import *
from CAB.cab_replay import *

if __name__ == '__main__': 
    
    directory = '/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/baseline_test/configs'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"processing {file}")
                    print(config)
                    
                    gen_cab(config)
                    replay_cab(config)
                    
                    gen_stitcher_plan(config)
                    do_stitcher_replay(config)