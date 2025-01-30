import json
import os
import re
import time
from datetime import datetime

from common.prometheus import prometheus_queries
from record_time import get_time
from databend_py import Client
from dotenv import load_dotenv


def load_config():
    """ Load configuration from environment variable. """
    load_dotenv()
    return {
        "host": os.getenv("HOST"),
        "databend_port": os.getenv("DATABEND_PORT"),
        "prometheus_port": os.getenv("PROMETHEUS_PORT"),
        "query": os.getenv("LP_QUERY_SET", "").split(","),
        "db": os.getenv("LP_DATABASE", "").split(","),
        "wait": int(os.getenv("WAIT_TIME", 0))
    }


def load_query_set(query_set):
    """ Load SQL statements from a file. """
    sql_statements = []
    query_path = os.path.join(os.path.dirname(__file__), "query", f"{query_set}.sql")
    current_statement = ""
    sort_key = None

    with open(query_path, "r") as file:
        for line in file:
            if line.startswith(" '.SQL/"):
                if current_statement:
                    sql_statements.append((sort_key, current_statement.strip()))
                    current_statement = ""
                sort_key = int(re.search(r"\.SQL/(\d+)\.0", line).group(1))
            else:
                current_statement += line.strip() + " "

    if current_statement:
        sql_statements.append((sort_key, current_statement.strip()))
    # sql_statements.sort(key=lambda x: x[0])
    sorted_sql_statements = [stmt[1] for stmt in sql_statements]
    
    return sorted_sql_statements

def load_query_from_json(path):
    with open(path, "r") as file:
        return json.load(file)
    

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
        print(q)
        try:
            tmp = client.execute(q)
            ret.append(tmp)
        except Exception as e:
            print(f"Error: {e}")
            pass
    return ret

def record_metrics(host, databend_port, prometheus_port, query, wait_time,database):
    """ Record and print metrics related to the executed query. """
    # start_time = time.time()
    start_time = get_time()
    
    print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    start_cputime = prometheus_queries["cpu"](host, prometheus_port, start_time)
    start_scan = prometheus_queries["scan"](host, prometheus_port, start_time)

    execute_query(host, databend_port, query,database)
    time.sleep(wait_time)

    # end_time = time.time()
    end_time = get_time()
    print(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    end_cputime = prometheus_queries["cpu"](host, prometheus_port, end_time)
    end_scan = prometheus_queries["scan"](host, prometheus_port, end_time)

    return query, end_cputime - start_cputime, end_scan - start_scan, end_time - start_time - wait_time


def save_data_to_file(data, record_file):
    """ Save data to a file. """
    with open(record_file, "w") as file:
        json.dump(data, file, indent=2)


def record_operator(host, databend_port, query,database):
    """ Record the operators used in the query. """
    operator_keywords = {
        "filter": "Filter",
        "join": "HashJoin",
        "agg": "AggregateFinal",
        "sort": "Sort"
    } # TODO: These are the keywords for the operators in the Databend execution plan. Update them as needed.
    plan = execute_query(host, databend_port, query,database)
    operator_flag = {}
    for i in range(len(plan)):
        tmp = '\n'.join([row[0] for row in plan[i][2]])
        for operator, keyword in operator_keywords.items():
            if keyword in tmp:
                operator_flag[operator] = 1
            else:
                operator_flag[operator] = 0
    return operator_flag


def main():
    config = load_config()
    
    record_file = "/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/one_last_exp/metrics_witho/llm-llm-sql-metrics.json"
    src = "/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/one_last_exp/metrics_witho/llm1-llm1-sql-metrics.json"
    sql_statements = load_query_from_json(src)
    data = []

    if os.path.exists(record_file):
        with open(record_file, "r") as file:
            data = json.load(file)
    start_index = len(data)

    for sql in sql_statements[start_index:]:
        query = sql["query"]
        query, database = query.split("@")
        # TODO: Explain Analyze or not?
        operaters = record_operator(config["host"], config["databend_port"], query, database)
        total_cputime, total_scan, total_duration = 0, 0, 0
        time.sleep(config["wait"])
        repeat=3
        for _ in range(repeat):
            query, cputime, scan, duration = record_metrics(config["host"], config["databend_port"], config["prometheus_port"], query, config["wait"],database)
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

        data.append({
            "query": query + "@" + database,
            "avg_cpu_time": total_cputime,
            "avg_scan_bytes": total_scan,
            "avg_duration": total_duration
            ,**operaters
        })
        save_data_to_file(data, record_file)


if __name__ == "__main__":
    main()
