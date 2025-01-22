import os

import sys
import subprocess
import time
from func_timeout import func_set_timeout
import func_timeout

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
sys.path.append("/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/common")
from sys import api_version
from dotenv import load_dotenv
import pandas as pd
import json
from copy import deepcopy
import numpy as np
from datetime import datetime
import os, copy, time
from openai import OpenAI
from databend_py import Client

import sys
sys.path.append("/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush")
from prometheus import prometheus_queries

def get_time():
    # get local time
    timestamp = time.time()
                        
    return timestamp


# 加入读args的包
import argparse

import signal

import platform


class TimeoutException(Exception):
    pass


if platform.system() == "Windows":
    import threading

    def timeout_handler():
        raise TimeoutException("Query execution timed out")

else:
    import signal

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Query execution timed out")


def cosine_similarity(vec1, vec2):
    # 将输入转换为 numpy 数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 计算向量的点积
    dot_product = np.dot(vec1, vec2)

    # 计算向量的模
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 计算余弦相似度
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # 防止除以零
    else:
        return dot_product / (norm_vec1 * norm_vec2)

def read_workload(config):
    with open(config["workload_path"], "r") as f:
        workload_df = pd.read_csv(f)
    return workload_df


class Query:
    def __init__(
        self,
        database,
        text,
        cpu,
        s_cpu,
        scan,
        s_scan,
        join,
        agg,
        duration,
        s_join=0,
        s_agg=0,
        loop_num=0,
        in_loop_num=0,
        is_valid=1,
    ) -> None:
        self.database = database
        self.text = text
        self.cpu = cpu
        self.s_cpu = s_cpu
        self.scan = scan
        self.s_scan = s_scan
        self.join = join
        self.agg = agg
        self.duration = duration
        self.s_join = s_join
        self.s_agg = s_agg
        self.loop_num = loop_num
        self.in_loop_num = in_loop_num
        self.is_valid = is_valid

    def cal_distance(self, other):
        a = [self.cpu, self.scan, self.join, self.agg]
        b = [other.cpu, other.scan, other.join, other.agg]
        return cosine_similarity(a, b)

    def check_same(self,other):
        if self.text == other.text and self.database == other.database:
            return True
        return False
    
    def describe_difference(self, target_query, describe_performance=False):
        prompt = [f"The query you have generated is as follows: {self.text}"]
        has_hasnt = ["hasn't", "has"]
        prompt.append(
            f"The query you have generated has {self.join} join operator, {self.agg} aggregation operator."
        )
        prompt.append(
            f"The query you need to generate has {target_query.join} join operator, {target_query.agg} aggregation operator."
        )
        if describe_performance:
            prompt.append(
                f"The query you have generated consumes {self.cpu} seconds of CPU time and scans {self.scan} GB of data."
            )
            prompt.append(
                f"The query you need to generate consumes {target_query.cpu} seconds of CPU time and scans {target_query.scan} GB of data."
            )
        if describe_performance:
            if self.join > target_query.join:
                prompt.append(f"Try to generate a query that has fewer join operators.")
            else:
                prompt.append(f"Try to generate a query that has more join operators.")
            if self.agg > target_query.agg:
                prompt.append(f"Try to generate a query that has fewer aggregation operators.")
            else:
                prompt.append(f"Try to generate a query that has more aggregation operators.")
            if self.cpu > target_query.cpu:
                prompt.append(f"Try to generate a query that consumes less CPU time.")
            else:
                prompt.append(f"Try to generate a query that consumes more CPU time.")
                prompt.append(create_more_cpu_hint())
            if self.scan > target_query.scan:
                prompt.append(f"Try to generate a query that scans less data.")
            else:
                prompt.append(f"Try to generate a query that scans more data.")
                prompt.append(create_more_scan_hint())
        return "\n".join(prompt)

    def refresh_syntax(self):
        self.text = self.text.replace("```sql", "").replace("```", "")
        if self.text.find("SELECT") == -1:
            self.is_valid = 0
        # if no ";" in the query, add one
        if self.text.find(";") == -1:
            self.text = self.text + ";"
        self.text = self.text[self.text.find("SELECT") : self.text.rfind(";") + 1]
        self.text = self.text.replace("\n", " ")

    def replay_and_fetch(self, config, repeat):
        print(f"Replaying Query: {self.text}")
        replay_log = ""
        if not self.is_valid:
            return 0
        operators = record_operator(
            config["host"], config["databend_port"], self.text, self.database
        )
        if operators["agg"] == -1:
            self.is_valid = 0
            return 0
        total_cputime, total_scan, total_duration = 0, 0, 0
        time.sleep(config["wait"])
        if repeat == 0:
            return 1
        for _ in range(repeat):
            query, cputime, scan, duration = record_metrics(
                config["host"],
                config["databend_port"],
                config["prometheus_port"],
                self.text,
                config["wait"],
                database=self.database,
            )
            if cputime < 0:
                self.is_valid = 0
                return 0
            print(cputime)
            total_cputime = total_cputime + cputime
            total_scan = total_scan + scan
            total_duration = total_duration + duration
            replay_log += f"Total time: {duration}"
            replay_log += f"Total cputime: {cputime}"
            replay_log += f"Total scan: {scan}"
            replay_log += f"---------------------"
        # add replay log to a new line of replay_log.txt
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "output",
                "replay_log.txt",
            ),
            "a",
        ) as f:
            f.write(replay_log)
        total_cputime = total_cputime / repeat
        total_scan = total_scan / repeat
        total_duration = total_duration / repeat
        self.cpu = total_cputime
        self.scan = total_scan / (1024 ** 3)
        self.duration = total_duration
        self.join = operators["join"]
        self.agg = operators["agg"]
        return 1

    def save_query(self, output_path):
        self_json = {
            "database": self.database,
            "query": self.text+ "@"+self.database,
            "avg_cpu_time": self.cpu,
            "s_cpu": self.s_cpu,
            "avg_scan_bytes": self.scan*(1024**3),
            "s_scan": self.s_scan*(1024**3),
            "avg_duration": self.duration,
            # TypeError: Object of type int32 is not JSON serializable
            "join": int(self.join),
            "agg": int(self.agg),
            "s_join": int(self.s_join),
            "s_agg": int(self.s_agg),
            "loop_num": int(self.loop_num),
            "in_loop_num": int(self.in_loop_num),
            "is_valid": int(self.is_valid),
        }
        # if not exist, create the file
        if not os.path.exists(
            os.path.join(
                output_path
            )
        ):
            with open(
                os.path.join(
                    output_path
                ),
                "w",
            ) as f:
                json.dump([self_json], f, indent=4)
        else:
            with open(
                os.path.join(
                    output_path
                ),
                "r+",
            ) as f:
                data = json.load(f)
                data.append(self_json)
                # clear the file
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=4)


def load_config():
    """Load configuration from environment variable."""
    load_dotenv()
    return {
        "workload_path": os.getenv("WORKLOAD_PATH"),
        "query": os.getenv("LP_QUERY_SET", "").split(","),
        "db": os.getenv("LP_DATABASE", "").split(","),
        "count_limit": int(os.getenv("LP_COUNT_LIMIT", 0)),
        "time_limit": int(os.getenv("LP_TIME_LIMIT", 0)),
        "host": os.getenv("HOST"),
        "databend_port": os.getenv("DATABEND_PORT"),
        "wait": int(os.getenv("WAIT_TIME", 0)),
        "prometheus_port": os.getenv("PROMETHEUS_PORT"),
        "ssh_command": os.getenv("SSH_COMMAND"),
    }


def read_sql_records(query_set, database):
    """Read SQL records from a JSON file."""
    record_file = os.path.join(
        "/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/one_last_exp/metrics_witho",
        f"{query_set}-{database}-sql-metrics.json",
    )
    with open(record_file, "r") as f:
        return json.load(f)


def find_k_nearest_neighbors(pool, virtual_query, k):
    ret = sorted(
        deepcopy(pool),
        key=lambda query: query.cal_distance(virtual_query),
        reverse=True,
    )
    return ret[:k]


def find_k_distants_neighbors(pool, virtual_query, k):
    ret = sorted(deepcopy(pool), key=lambda query: query.cal_distance(virtual_query))
    return ret[:k]


def create_positive_pool(new_query_list):
    sql_candidates = []
    database_set = ["tpch500m","tpch1g","tpch5g","tpch9g","llm"]
    query_sets = ["TPCH","TPCH","TPCH","TPCH","llm"]
    for query_set, database in zip(query_sets, database_set):
        records = read_sql_records(query_set, database)
        for record in records:
            if record["avg_cpu_time"] <0.1:
                continue
            sql_candidates.append(
                Query(
                    database = (record["database"] if database=="llm" else database),
                    text=record["query"],
                    cpu=record["avg_cpu_time"],
                    s_cpu=record["avg_cpu_time"],
                    scan=record["avg_scan_bytes"] / (1024**3),
                    s_scan=record["avg_scan_bytes"] / (1024**3),
                    join = record["join"],
                    agg = record["agg"],
                    duration=record["avg_duration"],
                    s_join=record["join"],
                    s_agg=record["agg"],
                    loop_num=0,
                    in_loop_num=0,
                    is_valid=1,
                )
            )
    sql_candidates.extend(new_query_list)
    return sql_candidates

def create_negative_pool(new_query_list):
    return create_positive_pool(new_query_list)


### Demo End ###

### OpenAI Begin ###

class Llm:
    def __init__(
        self,
        key_file=os.path.join(os.path.dirname(__file__), "..", "input", "keys.txt"),
        model="gpt-4o",
        temp=0.1,
        print_key=False,
    ):
        self.key_file = key_file
        self.model = model
        self.temp = temp
        self.client = None
        self.print_key = print_key

    def get_key(self):
        with open(self.key_file, "r") as f:
            keys = [x.strip() for x in f.readlines()]
        cur_key = copy.deepcopy(keys[0])
        keys = keys[1:] + [cur_key]
        self.cur_key = cur_key
        return cur_key

    def query(self, ask, get_lower=False):
        try:
            return self._query(ask, get_lower)
        except Exception as e:
            print(f"Error: {e}")
            if "maximum context length" in str(e):
                raise ValueError(
                    f"E(GPT): Maximum context length exceeded. Please reduce the input length."
                )
            if "You exceeded your current quota" in str(e):
                print("!!!!!!!!!!!!!!!! Please change the key file !!!!!!!!!!!!!!!!")
                time.sleep(60 * 1)
            # time.sleep(2)
            return self.query(ask, get_lower)

    def query_concate(self, ask_list, get_lower=False):
        try:
            return self._query_concate(ask_list, get_lower)
        except Exception as e:
            print(f"Error: {e}")
            if "maximum context length" in str(e):
                raise ValueError(
                    f"E(GPT): Maximum context length exceeded. Please reduce the input length."
                )
            if "You exceeded your current quota" in str(e):
                print("!!!!!!!!!!!!!!!! Please change the key file !!!!!!!!!!!!!!!!")
                time.sleep(60 * 1)
            # time.sleep(2)
            return self.query_concate(ask_list, get_lower)

    def _query_concate(self, ask_list, get_lower=False):
        key = self.get_key()
        if self.print_key:
            print(f"cur_key: {key}")
        os.environ["OPENAI_API_KEY"] = key
        if self.client is None:
            self.client = OpenAI(base_url="https://35.aigcbest.top/v1")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You can only generate a SQL that can be executed on TPC-H database. Except for the query itself, DO NOT generate other words",
            }
        ]
        turn = 0
        for ask in ask_list:
            if turn % 2 == 0:
                messages.append({"role": "user", "content": ask})
            else:
                messages.append({"role": "assistant", "content": ask})
            turn += 1
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temp if self.temp != -1 else 1,
            max_tokens=2048,
        )
        ans = completion.choices[0].message.content
        if get_lower:
            ans = ans.lower().strip().replace("\n", " ").replace("  ", " ")
        else:
            ans = ans.strip().replace("\n", " ").replace("  ", " ")
        return ans

    def _query(self, ask, post_process=False, get_lower=False):
        key = self.get_key()
        if self.print_key:
            print(f"cur_key: {key}")
        os.environ["OPENAI_API_KEY"] = key
        if self.client is None:
            self.client = OpenAI(base_url="https://35.aigcbest.top/v1")
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You can only generate a SQL that can be executed on TPC-H database. Except for the query itself, DO NOT generate other words",
                },
                {"role": "user", "content": ask},
            ],
            temperature=self.temp if self.temp != -1 else 1,
            max_tokens=2048,
        )
        ans = completion.choices[0].message.content
        if post_process:
            if get_lower:
                ans = ans.lower().strip().replace("\n", " ").replace("  ", " ")
            else:
                ans = ans.strip().replace("\n", " ").replace("  ", " ")
        return ans


### OpenAI End ###

### Replay Begin ###

@func_set_timeout(80)
def execute_query(host, port, query, database):
    # if there are more than one query in the file, only execute them one by one
    query = query.split(";")
    # make sure the last query is not empty
    if query[-1] == "":
        query = query[:-1]
    # add the last semicolon to each query
    query = [q + ";" for q in query]
    ret = []
    for q in query:
        if not q.startswith("Explain Analyze") and not q.startswith("EXPLAIN ANALYZE"):
            q = "Explain Analyze " + q
        client = Client(f"root:@{host}", port=port, secure=False, database=database)
        ret.append(client.execute(q))
    return ret


def record_operator(host, databend_port, query, database):
    dic = {
        "filter": ["Filter"],
        "join": ["HashJoin", "MergeJoin"],
        "agg": ["AggregateFinal"],
        "sort": ["Sort"],
    }
    """Record the operators used in the query."""
    try:
        plan = execute_query(host, databend_port, query, database)
    except func_timeout.exceptions.FunctionTimedOut:
        print(f"Error: ")
        return {"filter": -1, "join": -1, "agg": -1, "sort": -1}
    except Exception as e:
        print(f"Error: ")
        return {"filter": -1, "join": -1, "agg": -1, "sort": -1}
    operator_cnt = {"filter": 0, "join": 0, "agg": 0, "sort": 0}
    # transform the plan to a string
    plan = str(plan)
    for operator in dic:
        for op in dic[operator]:
            operator_cnt[operator] += plan.count(op)
    
    return operator_cnt


def record_metrics(host, databend_port, prometheus_port, query, wait_time, database):
    """Record and print metrics related to the executed query."""
    start_time = get_time()
    print(
        f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    start_cputime = prometheus_queries["cpu_new"](host, prometheus_port, start_time)
    start_scan = prometheus_queries["scan"](host, prometheus_port, start_time)
    try:
        execute_query(host, databend_port, query, database)
    except Exception as e:
        print(f"Error: {e}")
        return (
            query,
            -1,
            -1,
            -1,
        )
    time.sleep(wait_time)
    end_time = get_time()
    print(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    end_cputime = prometheus_queries["cpu_new"](host, prometheus_port, end_time)
    end_scan = prometheus_queries["scan"](host, prometheus_port, end_time)
    if (
        end_scan - start_scan < 1
        or end_cputime - start_cputime < 0
        or end_time - start_time < 0
    ):
        return (
            query,
            -1,
            -1,
            -1,
        )
    print(f"CPU time: {end_cputime - start_cputime}")
    print(f"Scan bytes: {end_scan - start_scan}")
    print(f"Duration: {end_time - start_time- wait_time}")
    return (
        query,
        end_cputime - start_cputime,
        end_scan - start_scan,
        end_time - start_time - wait_time,
    )


def create_database_option():
    database_vector = np.random.choice(["tpch500m", "tpch1g", "tpch5g","tpch9g"],p=[0.33,0.33,0.33,0.01])
    return database_vector


def create_database_guide_prompt(to_database):
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "input",
            "table_schema",
            "table_meta.json",
        )
    ) as f:
        schemas: list = json.load(f)
    for database in schemas:
        if database["database"] == to_database:
            return f"You are required to generate a SQL query on {to_database} database. the database schema and the table size of each table is {json.dumps(database['tables'])}. The query should have the following properties:"


def create_perf_goal(workload_df, is_perf=False, is_op=False, count_limit=50):
    row_id = np.random.randint(0, len(workload_df))
    target_query = workload_df.iloc[row_id]
    if is_perf:
        cpu_goal = target_query["cputime_sum"] / (count_limit / 2) * 2
        scan_goal = target_query["scanbytes_sum"] / (count_limit / 2)
    else:
        cpu_goal = 0
        scan_goal = 0
    if is_op:
        # the possiblity of filte-goal is 1 is the target_query["filter"]
        join_goal = np.random.choice(
            [0, 1], p=[1 - target_query["join"], target_query["join"]]
        )
        agg_goal = np.random.choice(
            [0, 1], p=[1 - target_query["agg"], target_query["agg"]]
        )
        op_vector = [join_goal, agg_goal]
        if op_vector == [0, 0, 0, 0]:
            if np.random.rand() < 0.4:
                op_vector = create_operator_vector()
    else:
        op_vector = create_operator_vector()
    return cpu_goal, scan_goal, op_vector


def create_perf_guide_prompt(target_query, is_perf=False):
    if is_perf:
        return f"The query should consume {target_query.cpu} seconds of CPU time and scan {target_query.scan} GB of data."
    else:
        return f"Please generate a SQL query that consumes as much CPU resources as possible while scanning as little data as possible. Please do not generate a SQL more than 20 operators in the query to avoid the query being too complex."


def create_operator_vector():
    choice = np.random.rand()
    op_vector = [np.random.choice([0, 1]) for i in range(4)]
    while sum(op_vector) >= 2 and np.random.rand() < 0.5:
        op_vector = [np.random.choice([0, 1]) for i in range(4)]
    return op_vector
    # return [1, 1, 1, 1]


def create_operator_guide_prompt(target_query):
    prompt = []
    operators = {
        "join": target_query.s_join,
        "agg": target_query.s_agg}
    for operator, count in operators.items():
        if count > 0:
            prompt.append(f"The query should contain {count} {operator} operators.")
        else:
            prompt.append(f"The query should not contain any {operator} operator.")
    return "\n".join(prompt)


def create_negative_prompt_hint(target_query, n_sql_candidates, k):
    hint_prompt = "There are some queries for you to refer to as negative examples, try to avoid there computing and scanning logic to match the target."
    n_sql_candidates = [
        record
        for record in n_sql_candidates
        if record.database == target_query.database
    ]
    n_final_examples = find_k_distants_neighbors(n_sql_candidates, target_query, k)
    for i, record in enumerate(n_final_examples):
        hint_prompt += f"{i+1}. \n SQL:    {record.text}    \n CPU Time: {record.cpu} \n Scan GBytes: {record.scan} \n Database: {record.database} \n"
    return hint_prompt, n_final_examples


def create_positive_prompt_hint(target_query, p_sql_candidates, k):
    hint_prompt = "There are some queries for you to refer to as positive examples, try to mimic there computing and scanning logic to match the target."
    p_sql_candidates = [
        record
        for record in p_sql_candidates
        if record.database == target_query.database
    ]
    p_final_examples = find_k_nearest_neighbors(p_sql_candidates, target_query, k)
    
    for i, record in enumerate(p_final_examples):
        hint_prompt += f"{i+1}. \n SQL:    {record.text}    \n CPU Time: {record.cpu} \n Scan GBytes: {record.scan} \n Database: {record.database} \n"
    return hint_prompt, p_final_examples

def create_more_cpu_hint():
    prompt="If you want to generate a query that consumes more CPU time, while scanning nearly the same amount of data, here are some hints for you: \n1. If you are able to add join operators, you can generate some self-join queries, this will significantly increase the intermediate data size and consume more CPU time. Examples: SELECT SIN(a.id) FROM table a JOIN table a ON a.id = a.id; \n 2. If you are required to generate a SQL without any operators or only have a few operators while having large amount of CPU time, you can generate a SQL like SELECT SELECT SIN(SIN(SIN(SIN(EXP(a))) FROM nation;\n 3. Try to generate a query that do some string operation on the data, like SUBSTRING, CONCAT, etc. Examples: SELECT CONCAT(a, b) FROM table; \n "
    return prompt
def create_more_scan_hint():
    prompt="If you want to generate a query that scans more data, while consuming nearly the same amount of CPU time, here are some hints for you: \n 1. Try to generate a query that do some full table scan. Examples: SELECT * FROM table; \n"
    return prompt

def try_to_restartdb_server():
    print("TRYING TO RESTART THE DATABASE SERVER")
    config = load_config()
    command = f'{config["ssh_command"]} "ps -ef | grep bend | grep -v grep | cut -c 9-15 |sudo xargs kill -9"'
    # run the command and no matter what the result is, run the next command
    command1 = subprocess.Popen(command, shell=True)
    command1.wait()
    # start the database server
    command = f'{config["ssh_command"]} "cd databend && sudo bash ./scripts/start.sh"'
    command2 = subprocess.Popen(command, shell=True)
    try:
        command2.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return 0


### Replay End ###
def test_connection():
    config = load_config()
    print("Current Time: ", get_time())
    test_query = Query(
        database="tpch500m",
        text="SELECT 1 FROM nation",
        cpu=0,
        s_cpu=0,
        scan=0,
        s_scan=0,
        join=0,
        agg=0,
        duration=0,
        s_join=0,
        s_agg=0,
        loop_num=0,
        in_loop_num=0,
        is_valid=1,
    )
    test_query.replay_and_fetch(config, 0)
    if not test_query.is_valid:
        print("CONNECTION TO DATABASE SERVER FAILED")
        test_query.is_valid = 1
        # try_to_restartdb_server()
        test_query.replay_and_fetch(config, 0)
        if not test_query.is_valid:
            print("RESTARTING DATABASE SERVER FAILED")
            return 0
        else:
            print("RESTARTING DATABASE SERVER SUCCESS")
            return 1
    else:
        return 1

def create_operator_vector_based_on_goal(join_goal,agg_goal):
    new_join_goal= np.random.choice([0, 1], p=[1 - join_goal, join_goal])
    new_agg_goal= np.random.choice([0, 1], p=[1 - agg_goal, agg_goal])
    return [new_join_goal,new_agg_goal]


def generate_query(config, cpu_total_goal, scan_total_goal,join_goal, agg_goal,max_loop=5, output_num=20, output_path="/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/LLM_new/output/llm-llm-sql-metrics.json", replay=3):
    print(cpu_total_goal,scan_total_goal)
    generated_queries=[]
    positive_pool=create_positive_pool(generated_queries)
    # set goal
    total_cpu=0
    total_scan=0
    total_num=0
    for query in positive_pool:
        total_cpu += query.cpu
        total_scan += query.scan
        total_num += 1
    avg_cpu = total_cpu / total_num
    avg_scan = total_scan / total_num
    cpu_scale=cpu_total_goal / avg_cpu
    scan_scale=scan_total_goal / avg_scan
    scale=(cpu_scale + scan_scale)
    cpu_goal=cpu_total_goal/scale
    scan_goal=scan_total_goal/scale
    cpu_goal=cpu_goal if cpu_goal>0 else 0
    scan_goal=scan_goal if scan_goal>0 else 0 
    agg_goal = int(agg_goal/scale)
    join_goal = int(join_goal/scale)
    print(f"Trying to generate a query has {cpu_goal} cpu time and {scan_goal} GB scan bytes")
    total_out_num = output_num
    while output_num > 0:
        current_prompt = []
        output_num -= 1
        this_database = create_database_option()
        # init query
        target_query = Query(
            database=this_database,
            text="TARGET_QUERY",
            cpu=cpu_goal,
            s_cpu=cpu_goal,
            scan=scan_goal,
            s_scan=scan_goal,
            join=join_goal,
            agg=agg_goal,
            duration=0,
            s_join=join_goal,
            s_agg=agg_goal,
            loop_num=total_out_num - output_num,
            in_loop_num=0,
            is_valid=0,
        )
        prompt = []
        prompt.append(create_database_guide_prompt(target_query.database))
        prompt.append(create_perf_guide_prompt(target_query, is_perf=True))
        prompt.append(create_operator_guide_prompt(target_query))
        # init the positive and negative examples
        positive_examples_num = 3
        negative_examples_num = 3
        pool_positive = create_positive_pool(generated_queries)
        pool_negative = create_negative_pool(generated_queries)
        hint_positive, p_records = create_positive_prompt_hint(
                target_query, pool_positive, positive_examples_num
            )
        prompt.append(hint_positive)
        hint_negative, n_records = create_negative_prompt_hint(
                target_query, pool_negative, negative_examples_num
            )
        prompt.append(hint_negative)
        prompt_text = "\n".join(prompt)
        print("--------PROMPT--------")
        print(prompt_text)
        print("-----END PROMPT-------")

        llm = Llm()
        query_text = llm.query(prompt_text)
        current_prompt.append(prompt_text)
        current_prompt.append(query_text)
        current_query = Query(
            database=this_database,
            text=query_text,
            cpu=0,
            s_cpu=target_query.cpu,
            scan=0,
            s_scan=target_query.scan,
            join=0,
            agg=0,
            duration=0,
            is_valid=1,
            loop_num=total_out_num - output_num,
            in_loop_num=0,
            s_join=target_query.s_join,
            s_agg=target_query.s_agg,
        )
        for query in generated_queries:
            if query.text == current_query.text and query.database == current_query.database:
                current_query.cpu = query.cpu
                current_query.scan = query.scan
                current_query.duration = query.duration
                current_query.join = query.join
                current_query.agg = query.agg
        if current_query.cpu>0:
            print("Found same query, skip collecting")
        else:
            current_query.refresh_syntax()
            if not current_query.is_valid:
                print("------------------")
                print("Invalid query syntax, please try again.")
                print(current_query.text)
                print("------------------")
            else:
                current_query.replay_and_fetch(config, repeat=replay)
                if not current_query.is_valid:
                    print("------------------")
                    print("Invalid query to execute, please try again.")
                    print(current_query.text)
                    print("------------------")
                    if not test_connection():
                        return 0
                else:
                    print("------------------")
                    print(f"Query generated: {current_query.text}")
                    print(f"|Operator|Scheduled|Generated|")
                    print(f"|---|---|---|")
                    print(f"|CPU|{target_query.cpu}|{current_query.cpu}|")
                    print(f"|Scan|{target_query.scan}|{current_query.scan}|")
                    print(f"|Duration|{target_query.duration}|{current_query.duration}|")
                    print(f"|Join|{current_query.s_join}|{current_query.join}|")
                    print(f"|Agg|{current_query.s_agg}|{current_query.agg}|")
                    print("------------------")
            generated_queries.append(current_query)
            current_query.save_query(output_path=output_path)
        now_max_loop = max_loop
        while now_max_loop > 0:
            now_max_loop -= 1
            new_prompt = current_query.describe_difference(
                target_query, describe_performance=True
            )
            print("-------PROMPT------")
            print("----END PROMPT------")
            print(new_prompt)
            print("----END PROMPT------")
            current_prompt.append(new_prompt)
            query_text = llm.query_concate(current_prompt)
            current_prompt.append(query_text)
            current_query = Query(
                database=this_database,
                text=query_text,
                cpu=0,
                s_cpu=target_query.cpu,
                scan=0,
                s_scan=target_query.scan,
                join=0,
                agg=0,
                duration=0,
                is_valid=1,
                loop_num=total_out_num - output_num,
                in_loop_num=max_loop - now_max_loop,
                s_join=target_query.s_join,
                s_agg=target_query.s_agg,
            )
            for query in generated_queries:
                if query.text == current_query.text and query.database == current_query.database:
                    current_query.cpu = query.cpu
                    current_query.scan = query.scan
                    current_query.duration = query.duration
                    current_query.join = query.join
                    current_query.agg = query.agg
            if current_query.cpu>0:
                print("Found same query, skip collecting")
            else:
                current_query.refresh_syntax()
                if not current_query.is_valid:
                    print("------------------")
                    print("Invalid query syntax, please try again.")
                    print(current_query.text)
                    print("------------------")
                else:
                    current_query.replay_and_fetch(config, replay)
                    if not current_query.is_valid:
                        print("------------------")
                        print("Invalid query to execute, please try again.")
                        print(current_query.text)
                        print("------------------")
                        if not test_connection():
                            return 0
                    else:
                        print("------------------")
                        print(f"Query generated: {current_query.text}")
                        print(f"|Operator|Scheduled|Generated|")
                        print(f"|---|---|---|")
                        print(f"|CPU|{target_query.cpu}|{current_query.cpu}|")
                        print(f"|Scan|{target_query.scan}|{current_query.scan}|")
                        print(
                            f"|Duration|{current_query.duration}|{target_query.duration}|"
                        )
                        print(f"|Join|{current_query.s_join}|{current_query.join}|")
                        print(f"|Agg|{current_query.s_agg}|{current_query.agg}|")
                        print("------------------")
                generated_queries.append(current_query)
                current_query.save_query(output_path)
    import pandas as pd
    import numpy as np
    with open(output_path) as f:
        data = pd.read_json(f)
    # drop row if is_valid is 0
    print(len(data))
    data = data[data['is_valid'] == 1]
    # 在query行 去重
    data = data.drop_duplicates(subset='query')
    print(len(data))
    # save to the original file
    data.to_json(output_path, orient='records')
   

def main():
    time.sleep(10)
    query = Query(
        text="select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from tpch5g.part, tpch5g.supplier, tpch5g.partsupp, tpch5g.nation, tpch5g.region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 46 and p_type like '%COPPER' and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'AFRICA' and ps_supplycost = ( select min(ps_supplycost) from tpch5g.partsupp, tpch5g.supplier, tpch5g.nation, tpch5g.region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'AFRICA' ) order by s_acctbal desc, n_name, s_name, p_partkey;",
        database="tpch5g",
        cpu=0,
        s_cpu=0,
        scan=0,
        s_scan=0,
        join=0,
        agg=0,
        duration=0,
        s_join=0,
        s_agg=0,
        loop_num=0,
        in_loop_num=0,
        is_valid=1
    )
    query.replay_and_fetch(load_config(), 2)
    print("CPU: ", query.cpu)
    print("SCAN: ", query.scan)
    print("Right CPU and SCAN: ", 1.87,328224916.0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=int, default=1)
    parser.add_argument("--negative", type=int, default=1)
    parser.add_argument("--max_loop", type=int, default=3)
    parser.add_argument("--output_num", type=int, default=100)
    parser.add_argument("--replay", type=int, default=3)
    output_path_default = f"additional_query_{datetime.now().strftime('%Y%m%d%H%M%S')}_{parser.parse_args().output_num}_{parser.parse_args().max_loop}_{parser.parse_args().positive}_{parser.parse_args().negative}.json"
    parser.add_argument("--output_path", type=str, default=output_path_default)
    output_num = parser.parse_args().output_num
    config = load_config()
    max_loop = parser.parse_args().max_loop
    # Test Connection 1
    if not test_connection():
        return 0
    else:
        print("TEST SUCCESS")
    # main loop
    generated_queries = []
    while output_num > 0:
        max_loop = parser.parse_args().max_loop
        current_prompt = []
        output_num -= 1
        this_database = create_database_option()
        # init query
        workload_df = read_workload(config)
        cpu_goal, scan_goal, op_vector = create_perf_goal(
            workload_df, is_perf=True, is_op=True, count_limit=config["count_limit"]
        )
        target_query = Query(
            database=this_database,
            text="TARGET_QUERY",
            cpu=cpu_goal,
            s_cpu=cpu_goal,
            scan=scan_goal,
            s_scan=scan_goal,
            filter=op_vector[0],
            join=op_vector[1],
            agg=op_vector[2],
            sort=op_vector[3],
            duration=0,
            s_filter=op_vector[0],
            s_join=op_vector[1],
            s_agg=op_vector[2],
            s_sort=op_vector[3],
            loop_num=parser.parse_args().output_num - output_num,
            in_loop_num=0,
            is_valid=0,
        )
        prompt = []
        prompt.append(create_database_guide_prompt(target_query.database))
        prompt.append(create_perf_guide_prompt(target_query, is_perf=True))
        prompt.append(create_operator_guide_prompt(target_query))
        # init the positive and negative examples
        positive_examples_num = 3
        negative_examples_num = 3
        if_positive = parser.parse_args().positive
        if_negative = parser.parse_args().negative
        pool_positive = create_positive_pool(generated_queries)
        pool_negative = create_negative_pool(generated_queries)
        if if_positive:
            hint_positive, p_records = create_positive_prompt_hint(
                target_query, pool_positive, positive_examples_num
            )
            prompt.append(hint_positive)
        if if_negative:
            hint_negative, n_records = create_negative_prompt_hint(
                target_query, pool_negative, negative_examples_num
            )
            prompt.append(hint_negative)
        prompt_text = "\n".join(prompt)
        print("--------PROMPT--------")
        print(prompt_text)
        print("-----END PROMPT-------")

        llm = Llm()
        query_text = llm.query(prompt_text)
        current_prompt.append(prompt_text)
        current_prompt.append(query_text)
        current_query = Query(
            database=this_database,
            text=query_text,
            cpu=0,
            s_cpu=target_query.cpu,
            scan=0,
            s_scan=target_query.scan,
            filter=0,
            join=0,
            agg=0,
            sort=0,
            duration=0,
            is_valid=1,
            loop_num=parser.parse_args().output_num - output_num,
            in_loop_num=0,
            s_filter=target_query.s_filter,
            s_join=target_query.s_join,
            s_agg=target_query.s_agg,
            s_sort=target_query.s_sort,
        )
        for query in generated_queries:
            if query.text == current_query.text and query.database == current_query.database:
                current_query.cpu = query.cpu
                current_query.scan = query.scan
                current_query.duration = query.duration
                current_query.filter = query.filter
                current_query.join = query.join
                current_query.agg = query.agg
                current_query.sort = query.sort
        if current_query.cpu>0:
            print("Found same query, skip collecting")
        else:
            current_query.refresh_syntax()
            if not current_query.is_valid:
                print("------------------")
                print("Invalid query syntax, please try again.")
                print(current_query.text)
                print("------------------")
            else:
                current_query.replay_and_fetch(config, parser.parse_args().replay)
                if not current_query.is_valid:
                    print("------------------")
                    print("Invalid query to execute, please try again.")
                    print(current_query.text)
                    print("------------------")
                    if not test_connection():
                        return 0
                else:
                    print("------------------")
                    print(f"Query generated: {current_query.text}")
                    print(f"|Operator|Scheduled|Generated|")
                    print(f"|---|---|---|")
                    print(f"|CPU|{target_query.cpu}|{current_query.cpu}|")
                    print(f"|Scan|{target_query.scan}|{current_query.scan}|")
                    print(f"|Duration|{target_query.duration}|{current_query.duration}|")
                    print(f"|Filter|{current_query.s_filter}|{current_query.filter}|")
                    print(f"|Join|{current_query.s_join}|{current_query.join}|")
                    print(f"|Agg|{current_query.s_agg}|{current_query.agg}|")
                    print(f"|Sort|{current_query.s_sort}|{current_query.sort}|")
                    print("------------------")
            generated_queries.append(current_query)
            current_query.save_query(parser.parse_args().output_path)
        while max_loop > 0:
            max_loop -= 1
            new_prompt = current_query.describe_difference(
                target_query, describe_performance=True
            )
            print("-------PROMPT------")
            print("----END PROMPT------")
            print(new_prompt)
            print("----END PROMPT------")
            current_prompt.append(new_prompt)
            query_text = llm.query_concate(current_prompt)
            current_prompt.append(query_text)
            current_query = Query(
                database=this_database,
                text=query_text,
                cpu=0,
                s_cpu=target_query.cpu,
                scan=0,
                s_scan=target_query.scan,
                filter=0,
                join=0,
                agg=0,
                sort=0,
                duration=0,
                is_valid=1,
                loop_num=parser.parse_args().output_num - output_num,
                in_loop_num=parser.parse_args().max_loop - max_loop,
                s_filter=target_query.s_filter,
                s_join=target_query.s_join,
                s_agg=target_query.s_agg,
                s_sort=target_query.s_sort,
            )
            for query in generated_queries:
                if query.text == current_query.text and query.database == current_query.database:
                    current_query.cpu = query.cpu
                    current_query.scan = query.scan
                    current_query.duration = query.duration
                    current_query.filter = query.filter
                    current_query.join = query.join
                    current_query.agg = query.agg
                    current_query.sort = query.sort
            if current_query.cpu>0:
                print("Found same query, skip collecting")
            else:
                current_query.refresh_syntax()
                if not current_query.is_valid:
                    print("------------------")
                    print("Invalid query syntax, please try again.")
                    print(current_query.text)
                    print("------------------")
                else:
                    current_query.replay_and_fetch(config, parser.parse_args().replay)
                    if not current_query.is_valid:
                        print("------------------")
                        print("Invalid query to execute, please try again.")
                        print(current_query.text)
                        print("------------------")
                        if not test_connection():
                            return 0
                    else:
                        print("------------------")
                        print(f"Query generated: {current_query.text}")
                        print(f"|Operator|Scheduled|Generated|")
                        print(f"|---|---|---|")
                        print(f"|CPU|{target_query.cpu}|{current_query.cpu}|")
                        print(f"|Scan|{target_query.scan}|{current_query.scan}|")
                        print(
                            f"|Duration|{current_query.duration}|{target_query.duration}|"
                        )
                        print(f"|Filter|{current_query.s_filter}|{current_query.filter}|")
                        print(f"|Join|{current_query.s_join}|{current_query.join}|")
                        print(f"|Agg|{current_query.s_agg}|{current_query.agg}|")
                        print(f"|Sort|{current_query.s_sort}|{current_query.sort}|")
                        print("------------------")
                generated_queries.append(current_query)
                current_query.save_query(parser.parse_args().output_path)
        


if __name__ == "__main__":
    main()
