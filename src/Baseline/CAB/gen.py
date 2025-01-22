import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

def load_config():
    """ Load configuration from environment variable. """
    load_dotenv()
    return {
        "workload_path": os.getenv("WORKLOAD_PATH"),
        "query": os.getenv("LP_QUERY_SET", "").split(","),
        "db": os.getenv("LP_DATABASE", "").split(","),
        "workload_name": os.getenv("WORKLOAD_NAME", "")
    }


def read_sql_records(query_set, database):
    """ Read SQL records from a JSON file. """
    record_file = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/linear/output/wzl_new_test/metrics_witho/{query_set}-{database}-sql-metrics.json"
    with open(record_file, "r") as f:
        return json.load(f)


def save_plan(config, results):
    """ Save the optimization plan to a JSON file. """
    workload_name = config["workload_name"]
    plan_path = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/baseline_test/cab/output/{workload_name}/{'+'.join(sorted(config['query']))}-plan.json"
    with open(plan_path, "a") as f:
        json.dump(results, f, indent=2)

def save_tmp_res(config, results):
    """ Save the optimization plan to a JSON file. """
    workload_name = config["workload_name"]
    plan_path = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/baseline_test/cab/tmp_res/{workload_name}/{'+'.join(sorted(config['query']))}-results.json"
    with open(plan_path,"a") as f:
        json.dump(results, f, indent=2)

def main():
    config = load_config()

    sql_candidates = []
    for query_set, database in zip(config["query"], config["db"]):
        records = read_sql_records(query_set, database)
        for record in records:
            record["query"] += "@" + database
            record["avg_scan_bytes"] = record["avg_scan_bytes"] / (1024 ** 3)
            sql_candidates.append(record)

    workload = pd.read_csv(config["workload_path"])
    workload_time_slots = list(zip(workload['cputime_sum'], workload['scanbytes_sum'], workload['filter'], workload['join'], workload['agg'], workload['sort']))

    results = []
    for i, (cpu_time, scan_bytes, filter, join, agg, sort) in enumerate(workload_time_slots):
        current_cpu_time = 0
        current_duration = 0
        sqls = {}
        querys = []
        while current_cpu_time < cpu_time and current_duration < 1000000000000: 
            # randomly select a SQL candidate
            sql = np.random.choice(sql_candidates)
            if sql["query"] in sqls:
                sqls[sql["query"]] += 1
            else:
                sqls[sql["query"]] = 1
            current_cpu_time += sql["avg_cpu_time"]
            current_duration += sql["avg_duration"]
            querys.append(sql["query"])


        filter_count = 0
        for candidate in sql_candidates:
            filter_count += candidate["filter"] * sqls.get(candidate["query"], 0)
        join_count = 0
        for candidate in sql_candidates:
            join_count += candidate["join"] * sqls.get(candidate["query"], 0)
        agg_count = 0
        for candidate in sql_candidates:
            agg_count += candidate["agg"] * sqls.get(candidate["query"], 0)
        sort_count = 0
        for candidate in sql_candidates:
            sort_count += candidate["sort"] * sqls.get(candidate["query"], 0)

        sql_count = sum(sqls.values())
        filter_ratio = filter_count / sql_count
        join_ratio = join_count / sql_count
        agg_ratio = agg_count / sql_count
        sort_ratio = sort_count / sql_count

        # Time Begin

        ms_per_slot = 5 * 60 * 1000
        query_count_in_slot = len(querys)
        rate_per_second = query_count_in_slot / (ms_per_slot / 1000.0)
        dist_s = np.random.exponential(scale=1/rate_per_second, size=(query_count_in_slot)) 

        slot_start = 0
        now_ms = slot_start

        for j in range(len(querys)):
            now_ms += dist_s[j] * 1000.0
            now_ms = (now_ms - slot_start) % ms_per_slot + slot_start
            querys[j] = {'timestamp': now_ms, 'query_id': querys[j]}

        # Time End
        
        results.append({
            "queries": querys,
            "operator_ratios": {
                "filter": filter_ratio,
                "join": join_ratio,
                "agg": agg_ratio,
                "sort": sort_ratio
            }
        })

    save_plan(config, results)
    
def gen_cab(config):
    sql_candidates = []
    for query_set, database in zip(config["query"], config["db"]):
        records = read_sql_records(query_set, database)
        for record in records:
            record["query"] += "@" + database
            record["avg_scan_bytes"] = record["avg_scan_bytes"] / (1024 ** 3)
            sql_candidates.append(record)

    workload = pd.read_csv(config["workload_path"])
    workload_time_slots = list(zip(workload['cputime_sum'], workload['scanbytes_sum'], workload['filter'], workload['join'], workload['agg'], workload['sort']))

    results = []
    tmp_res = []
    for i, (cpu_time, scan_bytes, filter, join, agg, sort) in enumerate(workload_time_slots):
        current_cpu_time = 0
        current_duration = 0
        current_scan = 0
        sqls = {}
        querys = []
        while current_cpu_time < cpu_time and current_duration < 1000000000000: 
            # randomly select a SQL candidate
            sql = np.random.choice(sql_candidates)
            if sql["query"] in sqls:
                sqls[sql["query"]] += 1
            else:
                sqls[sql["query"]] = 1
            current_cpu_time += sql["avg_cpu_time"]
            current_scan += sql["avg_scan_bytes"]
            current_duration += sql["avg_duration"]
            querys.append(sql["query"])
        print(current_scan)
        print(scan_bytes)
        print(abs(current_scan-scan_bytes))
        print("-------------")


        filter_count = 0
        for candidate in sql_candidates:
            filter_count += candidate["filter"] * sqls.get(candidate["query"], 0)
        join_count = 0
        for candidate in sql_candidates:
            join_count += candidate["join"] * sqls.get(candidate["query"], 0)
        agg_count = 0
        for candidate in sql_candidates:
            agg_count += candidate["agg"] * sqls.get(candidate["query"], 0)
        sort_count = 0
        for candidate in sql_candidates:
            sort_count += candidate["sort"] * sqls.get(candidate["query"], 0)

        sql_count = sum(sqls.values())
        if sql_count:
            filter_ratio = filter_count / sql_count
            join_ratio = join_count / sql_count
            agg_ratio = agg_count / sql_count
            sort_ratio = sort_count / sql_count
        else:
            filter_ratio = 0
            join_ratio = 0
            agg_ratio = 0
            sort_ratio = 0

        # Time Begin

        ms_per_slot = 5 * 60 * 1000
        query_count_in_slot = len(querys)
        if query_count_in_slot != 0:
            rate_per_second = query_count_in_slot / (ms_per_slot / 1000.0)
            dist_s = np.random.exponential(scale=1/rate_per_second, size=(query_count_in_slot)) 

            slot_start = 0
            now_ms = slot_start

            for j in range(len(querys)):
                now_ms += dist_s[j] * 1000.0
                now_ms = (now_ms - slot_start) % ms_per_slot + slot_start
                querys[j] = {'timestamp': now_ms, 'query_id': querys[j]}

            # Time End
        
        results.append({
            "queries": querys,
            "operator_ratios": {
                "filter": filter_ratio,
                "join": join_ratio,
                "agg": agg_ratio,
                "sort": sort_ratio
            }
        })

        tmp_res.append(
            {
                "idx": i,
                "cpu_time_sum": current_cpu_time,
                "scan_bytes_sum": current_scan,
                "filter": filter_ratio,
                "join": join_ratio,
                "agg": agg_ratio,
                "sort": sort_ratio
            }
        )
    
    # save_plan(config, results)
    save_tmp_res(config, tmp_res)