import os, copy, time
from openai import OpenAI
from src.tools.utils import open_json, save_json, cut_cottable_prompt
from global_values import *
if OPENAI_BASE_URL is not None:
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

class GPTPOOL:
    def __init__(self, key_file=KEY_FILE, model="gpt-3.5-turbo-0301", temp=LLM_HYPER_PARAMS['temperature'], root_dir = './tmp/gpt', record_log=False, print_key=False):
        self.key_file = key_file
        self.model = model
        self.temp = temp
        self.client = None
        self.root_dir = root_dir
        self.record_log = record_log
        if os.path.exists(os.path.join(root_dir, 'log.json')):
            self.log = open_json(os.path.join(root_dir, 'log.json'))
        else:
            self.log = {}
        self.print_key = print_key

    def get_key(self):
        with open(self.key_file, 'r') as f:
            keys = [x.strip() for x in f.readlines()]

        cur_key = copy.deepcopy(keys[0])
        keys = keys[1:]+[cur_key]

        with open(self.key_file, 'w') as f:
            f.write('\n'.join(keys))

        self.cur_key = cur_key

        return cur_key
    
    def query(self, ask, get_lower=False):
        try:
            return self._query(ask, get_lower)
        except Exception as e:
            print(f'Error: {e}')
            if 'maximum context length' in str(e):
                raise ValueError(f'E(GPT): Maximum context length exceeded. Please reduce the input length.')
            if 'You exceeded your current quota' in str(e):
                print('!!!!!!!!!!!!!!!! Please change the key file !!!!!!!!!!!!!!!!')
                time.sleep(60*1)

            # time.sleep(2)
            return self.query(ask, get_lower)

    def _query(self, ask, post_process=False, get_lower=False):
        key = self.get_key()
        if self.print_key:
            print(f'cur_key: {key}')
        os.environ["OPENAI_API_KEY"] = key
        if self.client is None:
            self.client = OpenAI()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": ask}],
            temperature=self.temp if self.temp != -1 else 1,
            max_tokens=MAX_OUTPUT_LIMIT
        )
        # print(completion)
        ans = completion.choices[0].message.content
        if post_process:
            if get_lower:
                ans = ans.lower().strip().replace('\n', ' ').replace('  ', ' ')
            else:
                ans = ans.strip().replace('\n', ' ').replace('  ', ' ')
        if self.record_log:
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.log[cur_time] = {'ask': ask, 'ans': ans}
            save_json(self.log, os.path.join(self.root_dir, 'log.json'))
        return ans
    

gpt = GPTPOOL(key_file='../input/keys')