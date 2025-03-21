import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from prometheus import prometheus_queries
from databend_py import Client
from dotenv import load_dotenv
import subprocess

import re

import sys

def get_time():
    # get local time
    timestamp = time.time()
                        
    return timestamp

def execute_query(host, port, query, database):
    """ Execute a given SQL query using the databend client. """
    client = Client(f"root:@{host}", port=port, secure=False, database=database)
    try:
        _ = client.execute(query)
    except Exception as e:
        print(f"Error: {e}")
        pass


def load_plan(config):
    """ Load the execution plan from a JSON file. """
    workload_name = config["workload_name"]
    plan_path = f"./output/{workload_name}/{'+'.join(sorted(config['query']))}-plan.json"
    with open(plan_path, "r") as f:
        return json.load(f)


def execute_plan(config, plans):
    """ Execute the SQL plan and collect metrics. """
    execution_data = []
    for idx, plan in enumerate(plans[:]):
        queries = plan["queries"]
        operator_ratios = plan["operator_ratios"]

        start_time, start_cputime, start_scan = record_start_time(config)
        execute_threads(config, queries)
        time.sleep(config["wait"])
        end_time, end_cputime, end_scan = record_end_time(config)

        cpu = []
        scan = []
        for t in range(int(start_time), int(start_time + 30 * 10 + config["interval"]), config["interval"]):
            cpu_time = prometheus_queries["cpu_new"](config["host"], config["prometheus_port"], t)
            scan_bytes = prometheus_queries["scan"](config["host"], config["prometheus_port"], t)
            cpu.append(cpu_time)
            scan.append(scan_bytes)

        cpu = [cpu[i] - cpu[i - 1] for i in range(1, len(cpu))]
        scan = [scan[i] - scan[i - 1] for i in range(1, len(scan))]
        scan = [s / (1024 ** 3) for s in scan]

        idx, cpu_time, scan_bytes, duration = collect_metrics(idx, start_time, end_time, start_cputime, end_cputime, start_scan, end_scan, config["wait"])
        execution_data.append({
            "idx": idx,
            "cpu_time_sum": cpu_time,
            "scan_bytes_sum": scan_bytes,
            "cpu_time_interval": cpu,
            "scan_time_interval": scan,
            "duration": duration,
            **operator_ratios
        })
        print(execution_data[-1])
        time.sleep(config["wait"])
    duration = [data["duration"] for data in execution_data]
    return execution_data, np.mean(duration)


def record_start_time(config):
    """ Record the start time and initial metrics. """
    start_time = get_time()
    start_cputime = prometheus_queries["cpu_new"](config["host"], config["prometheus_port"], start_time)
    start_scan = prometheus_queries["scan"](config["host"], config["prometheus_port"], start_time)
    print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    return start_time, start_cputime, start_scan


def execute_threads(config, sql_per_time_interval):
    """ Start threads for SQL query execution. """
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = []
        for queries in sql_per_time_interval:
            query, database = queries['query_id'].split("@")
            query = query.split(";")
            # make sure the last query is not empty
            if query[-1] == "":
                query = query[:-1]
            for i in range(len(query)):
                query[i] = re.sub(r"as\s'([^']+)'", r'as "\1"', query[i])
            # add the last semicolon to each query
            query = [q + ";" for q in query]
            for i in range(len(query)):
                q = query[i]
                if not q.startswith("Explain Analyze") and not q.startswith("EXPLAIN ANALYZE"):
                    query[i] = "Explain Analyze " + query[i]
            for q in query:
                future = executor.submit(execute_query, config["host"], config["databend_port"], q, database)
                futures.append(future)
        for future in futures:
            _ = future.result()



def record_end_time(config):
    """ Record the end time and final metrics. """
    end_time = get_time()
    print(f"End time: {datetime.fromtimestamp(end_time - config['wait']).strftime('%Y-%m-%d %H:%M:%S')}")
    end_cputime = prometheus_queries["cpu_new"](config["host"], config["prometheus_port"], end_time)
    end_scan = prometheus_queries["scan"](config["host"], config["prometheus_port"], end_time)
    return end_time, end_cputime, end_scan


def collect_metrics(idx, start_time, end_time, start_cputime, end_cputime, start_scan, end_scan, wait_time):
    """ Collect and return metrics for a single execution. """
    return idx, end_cputime - start_cputime, end_scan - start_scan, end_time - start_time - wait_time


def save_results(config, data):
    """ Save the execution results to a JSON file. """
    workload_name = config["workload_name"]
    result_path = f"./output/{workload_name}/{'+'.join(sorted(config['query']))}-results.json"
    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)
    
def replay_cab(config):
    plan = load_plan(config)
    results, avg_duration = execute_plan(config, plan)
    print(f"Average Duration: {avg_duration:.2f}s")
    save_results(config, results)