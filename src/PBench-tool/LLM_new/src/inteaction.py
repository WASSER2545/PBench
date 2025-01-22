from msilib import schema
from ntpath import join
import os
from sys import api_version
from turtle import distance
from dotenv import load_dotenv
import pandas as pd
import json
from copy import deepcopy
import numpy as np
from datetime import datetime
import os, copy, time
from openai import OpenAI
from databend_py import Client
from common.prometheus import prometheus_queries



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


class Query:

    def __init__(self, database, text, cpu, scan, filter, join, agg, sort, duration) -> None:
        self.database = database
        self.text = text
        self.cpu = cpu
        self.scan = scan
        self.filter = filter
        self.join = join
        self.agg = agg
        self.sort = sort
        self.duration = duration

    def cal_distance(self, other):
        a = [self.cpu, self.scan, self.filter, self.join, self.agg, self.sort]
        b = [other.cpu, other.scan, other.filter, other.join, other.agg, other.sort]
        return cosine_similarity(a, b)


def load_config():
    """ Load configuration from environment variable. """
    load_dotenv()
    return {
        "workload_path": os.getenv("WORKLOAD_PATH"),
        "query": os.getenv("LP_QUERY_SET", "").split(","),
        "db": os.getenv("LP_DATABASE", "").split(","),
        "count_limit": int(os.getenv("LP_COUNT_LIMIT", 0)),
        "time_limit": int(os.getenv("LP_TIME_LIMIT", 0))
    }


def read_sql_records(query_set, database):
    """ Read SQL records from a JSON file. """
    record_file = os.path.join("D:\codespace\python\FlexBench\simulator\linear", f"output\metrics_count_o\{query_set}-{database}-sql-metrics.json")
    with open(record_file, "r") as f:
        return json.load(f)


def create_pool(config):
    sql_candidates = []
    for query_set, database in zip(config["query"], config["db"]):
        records = read_sql_records(query_set, database)
        for record in records:
            sql_candidates.append(Query(
                database=database, text=record["query"], 
                cpu=record["avg_cpu_time"], scan=record["avg_scan_bytes"] / (1024 ** 3),
                filter=1 if record["filter"] > 0 else 0,
                join=1 if record["join"] > 0 else 0,
                agg=1 if record["agg"] > 0 else 0,
                sort=1 if record["sort"] > 0 else 0,
                duration=record["avg_duration"]
            ))
    return sql_candidates

### Demo Begin ###

def find_k_nearest_neighbors(pool, virtual_query, k):
    ret = sorted(deepcopy(pool), key=lambda query: query.cal_distance(virtual_query), reverse=True)
    return ret[:k]


def find_k_distants_neighbors(pool, virtual_query, k):
    ret = sorted(deepcopy(pool), key=lambda query: query.cal_distance(virtual_query))
    return ret[:k]


def describe_deference(neighbor_query: Query, virtual_query: Query):
    prompt = [neighbor_query.text]
    if neighbor_query.cpu > virtual_query.cpu:
        prompt.append(f"The query given above consumes a higher CPU time of {neighbor_query.cpu} seconds.")
    else:
        prompt.append(f"The query given above query consumes a lower CPU time of {neighbor_query.cpu} seconds.")
    if neighbor_query.scan > virtual_query.scan:
        prompt.append(f"The query given above scans more bytes, up to {neighbor_query.cpu} GB.")
    else:
        prompt.append(f"The query given above scans fewer bytes, only {neighbor_query.scan} GB.")
    return "\n".join(prompt)

### Demo End ###

### OpenAI Begin ###

class Llm:
    def __init__(self, key_file=os.path.join(os.path.dirname(__file__), "..", "input", "keys.txt"), 
                 model="gpt-3.5-turbo-0301", temp=0, print_key=False):
        self.key_file = key_file
        self.model = model
        self.temp = temp
        self.client = None
        self.print_key = print_key


    def get_key(self):
        with open(self.key_file, 'r') as f:
            keys = [x.strip() for x in f.readlines()]
        cur_key = copy.deepcopy(keys[0])
        keys = keys[1:]+[cur_key]
        self.cur_key = cur_key
        return cur_key

    
    def query(self, ask, get_lower=False):
        return "select 1"
        # TODO: check below
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
            max_tokens=2048
        )
        ans = completion.choices[0].message.content
        if post_process:
            if get_lower:
                ans = ans.lower().strip().replace('\n', ' ').replace('  ', ' ')
            else:
                ans = ans.strip().replace('\n', ' ').replace('  ', ' ')
        return ans

### OpenAI End ###

### Replay Begin ###

def execute_query(host, port, query):
    """ Execute a given SQL query using the databend client. """
    client = Client(f"root:@{host}", port=port, secure=False)
    return client.execute(query)


def record_operator(host, databend_port, query):
    """ Record the operators used in the query. """
    operator_keywords = {
        "filter": "Filter",
        "join": "HashJoin",
        "agg": "AggregateFinal",
        "sort": "Sort"
    } # TODO: These are the keywords for the operators in the Databend execution plan. Update them as needed.
    plan = execute_query(host, databend_port, query)
    plan = '\n'.join([row[0] for row in plan[2]])
    operator_flag = {}
    for operator, keyword in operator_keywords.items():
        if keyword in plan:
            operator_flag[operator] = 1
        else:
            operator_flag[operator] = 0
    return operator_flag


def record_metrics(host, databend_port, prometheus_port, query, wait_time):
    """ Record and print metrics related to the executed query. """
    start_time = time.time()
    print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    start_cputime = prometheus_queries["cpu"](host, prometheus_port, start_time)
    start_scan = prometheus_queries["scan"](host, prometheus_port, start_time)

    execute_query(host, databend_port, query)
    time.sleep(wait_time)

    end_time = time.time()
    print(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    end_cputime = prometheus_queries["cpu"](host, prometheus_port, end_time)
    end_scan = prometheus_queries["scan"](host, prometheus_port, end_time)

    return query, end_cputime - start_cputime, end_scan - start_scan, end_time - start_time - wait_time


def replay_and_fetch(config, query, repeat=3):
    return {
        "query": query,
        "cpu": 1,
        "scan": 1,
        "duration": 1,
        "filter": 1,
        "join": 1,
        "agg": 1,
        "sort": 1,
    }
    # TODO: check below
    operaters = record_operator(config["host"], config["databend_port"], query)
    total_cputime, total_scan, total_duration = 0, 0, 0
    time.sleep(config["wait"])
    for _ in range(repeat):
        query, cputime, scan, duration = record_metrics(config["host"], config["databend_port"], config["prometheus_port"], query, config["wait"])
        total_cputime = total_cputime + cputime
        total_scan = total_scan + scan
        total_duration = total_duration + duration
        print(f"Total time: {duration}")
        print(f"Total cputime: {cputime}")
        print(f"Total scan: {scan}")
        print("-" * os.get_terminal_size().columns)
    total_cputime = total_cputime / repeat
    total_scan = total_scan / repeat
    total_duration = total_duration / repeat

    return {
        "query": query,
        "cpu": total_cputime,
        "scan": total_scan,
        "duration": total_duration,
        **operaters
    }

### Replay End ###

def main():
    k = 5
    max_loop = 3

    with open(os.path.join(os.path.dirname(__file__), "..", "input", "table_schema", "table_meta.json")) as f:
        schemas: list = json.load(f)

    config = load_config()
    pool=create_pool(config)
    # TODO: change the virtual query
    virtual_query = pool[-1]
    k_nearest_neighbors=find_k_nearest_neighbors(pool, virtual_query, k)
    k_distants_neighbors=find_k_distants_neighbors(pool, virtual_query, k)

    prompt = []
    for database in schemas:
        if database["database"] == virtual_query.database:
            prompt.append(f"You are required to generate a SQL query on {virtual_query.database} database. the database schema and the table size of each table is {json.dumps(database['tables'])}. The query should have the following properties:")
            break
    prompt.append(f"1. The average CPU time of the query should be around {virtual_query.cpu} seconds.")
    prompt.append(f"2. The average scan bytes of the query should be around {virtual_query.scan} GB.")
    prompt.append(f"3. The average duration of the query should be around {virtual_query.duration} seconds.")

    operators = {
        "filter": virtual_query.filter,
        "join": virtual_query.join,
        "agg": virtual_query.agg,
        "sort": virtual_query.sort,
    }
    for operator, count in operators.items():
        if count > 0:
            prompt.append(f"{len(prompt)}. The query should contain {operator} operator.")
        else:
            prompt.append(f"{len(prompt)}. The query should not contain any {operator} operator.")

    # TODO: Attention please, the index has been adjusted
    prompt.append("There are some queries that similar to the query you need to generate:")
    for i in range(len(k_nearest_neighbors)):
        prompt.append(f"{len(prompt) - 1}. {describe_deference(k_nearest_neighbors[i], virtual_query)}")
    
    prompt.append("There are some queries that very different from the query you need to generate:")
    for i in range(len(k_distants_neighbors)):
        prompt.append(f"{len(prompt) - 2}. {describe_deference(k_nearest_neighbors[i], virtual_query)}")

    prompt_text = "\n".join(prompt)


    llm = Llm()
    generated_query = []

    query = llm.query(prompt_text)
    metrics = replay_and_fetch(config, query)
    generated_query.append(Query(virtual_query.database, query, metrics["cpu"], metrics["scan"],
                                metrics["filter"], metrics["join"], metrics["agg"],
                                metrics["sort"], metrics["duration"]))
    prompt.append("There are some queries you have generated:")
    prompt.append(f"{len(prompt) - 3}. {describe_deference(generated_query[-1], virtual_query)}")
    while max_loop>0:
        prompt_text = "\n".join(prompt)
        query = llm.query(prompt_text)
        metrics = replay_and_fetch(config, query)
        generated_query.append(Query(virtual_query.database, query, metrics["cpu"], metrics["scan"],
                                    metrics["filter"], metrics["join"], metrics["agg"],
                                    metrics["sort"], metrics["duration"]))
        prompt.append(f"{len(prompt) - 3}. {describe_deference(generated_query[-1], virtual_query)}")
        max_loop-=1
    print("11")


if __name__ == "__main__":
    main()