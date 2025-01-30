import json
import os
import random

import numpy as np
import pandas as pd
from simanneal import Annealer

cpu_e = 0
scan_e = 0
cpu_es = []
scan_es = []
filter_error = []
sort_error = []
agg_error = []
join_error = []

def read_sql_records(query_set, database):
    """ Read SQL records from a JSON file. """
    record_file = os.path.join(f"../Collect_metrics/metrics_witho/output/{query_set}-{database}-sql-metrics.json")
    with open(record_file, "r") as f:
        return json.load(f)


class Problem(Annealer):

    def __init__(self, state, queries, cpu_target, scanbytes_target, 
                 filter_target, sort_target, join_target, agg_target):
        super(Problem, self).__init__(state)  # important!
        self.queries = queries
        self.cpu_target = cpu_target
        self.scanbytes_target = scanbytes_target
        self.cpu_base = np.mean(self.cpu_target)
        self.scanbytes_base = np.mean(self.scanbytes_target)
        self.filter_target = filter_target
        self.sort_target = sort_target
        self.join_target = join_target
        self.agg_target = agg_target
        self.interval = queries[0]["info"]["interval"]

    def calc_avg_duration(self, durations:list):
        if len(durations)==0:
            return 0
        if len(durations)==1:
            return durations[0]
        sorted_durations = sorted(durations)
        desc_durations = sorted_durations[::-1]
        power_list=[]
        for i in range(len(desc_durations)):
            if i==0:
                power_list.append(0)
            else:
                power_list.append((i + 1) / i)
        sorted_power_list = sorted(power_list)
        max_avg = 0
        min_avg = 0
        for i in range(len(sorted_power_list)):
            max_avg += sorted_durations[i] * sorted_power_list[i]
            min_avg += desc_durations[i] * (1 - sorted_power_list[i])
        max_avg /= len(sorted_power_list)
        min_avg /= len(sorted_power_list)
        return (max_avg+min_avg) / 2

    def reallocate_duration(self, lasted_durations:list, this_interval_duration:list):
        all_durations = []
        all_durations.extend(lasted_durations)
        all_durations.extend(this_interval_duration)
        avg_duration = self.calc_avg_duration(all_durations)
        total_duration = avg_duration * len(this_interval_duration)
        total_duration -= sum(lasted_durations)
        new_this_interval_duration = []
        for i in range(len(this_interval_duration)):
            new_this_interval_duration.append(total_duration * this_interval_duration[i] / sum(this_interval_duration))
        return new_this_interval_duration

    def reload_queries(self):
        all_intervals = [[] for _ in range(len(self.cpu_target))]
        for i, s in enumerate(self.state):
            all_intervals[s].append(i)
        last = []
        new_durations = [0] * len(self.queries)
        for qs in all_intervals:
            tmp = [self.queries[i]["info"]["avg_duration"] for i in qs]
            interval_new_duration = self.reallocate_duration(last, tmp)
            add = []
            for i in range(len(interval_new_duration)):
                new_durations[qs[i]] = interval_new_duration[i]
                if interval_new_duration[i] > self.interval:
                    last.append(interval_new_duration[i] - self.interval)
            while len(last) > 0:
                du = last.pop()
                if du > self.interval:
                    add.append(du - self.interval)
            last.extend(add)
        return new_durations
    def move(self):
        initial_energy = self.energy()
        query_idx = random.randint(0, len(self.queries) - 1)
        self.state[query_idx] = random.randint(0, len(self.cpu_target) - self.queries[query_idx]["info"]["avg_lasted"])
        return self.energy() - initial_energy

    def energy(self):
        workload_cpu = [0.0 for _ in range(len(self.cpu_target))]
        workload_scanbytes = [0.0 for _ in range(len(self.scanbytes_target))]
        workload_filter = [0.0 for _ in range(len(self.filter_target))]
        workload_sort = [0.0 for _ in range(len(self.sort_target))]
        workload_join = [0.0 for _ in range(len(self.join_target))]
        workload_agg = [0.0 for _ in range(len(self.agg_target))]
        workload_sql_cnt = [0 for _ in range(len(self.cpu_target))]
        
        new_durations = self.reload_queries()
        
        for i, query in enumerate(self.queries):
            interval = self.interval
            cpu_duration = new_durations[i]
            cpu_per_second = query["info"]["avg_cpu_time"] / cpu_duration
            query_lasted = int(cpu_duration // interval + 1)
            workload_scanbytes[self.state[i] + query_lasted - 1] += query["info"]["avg_scan_bytes"]
            for j in range(query_lasted):
                # cputime
                if cpu_duration >= interval:
                    workload_cpu[self.state[i] + j] += cpu_per_second * interval
                else:
                    workload_cpu[self.state[i] + j] += cpu_per_second * cpu_duration
                cpu_duration -= interval
                
                # operators
                workload_filter[self.state[i] + j] += query["info"]["filter"]
                workload_sort[self.state[i] + j] += query["info"]["sort"]
                workload_join[self.state[i] + j] += query["info"]["join"]
                workload_agg[self.state[i] + j] += query["info"]["agg"]
                workload_sql_cnt[self.state[i] + j] += 1
        
        for i in range(len(self.filter_target)):
            if workload_sql_cnt[i]:
                workload_filter[i] /= workload_sql_cnt[i]
                workload_sort[i] /= workload_sql_cnt[i]
                workload_join[i] /= workload_sql_cnt[i]
                workload_agg[i] /= workload_sql_cnt[i]
                
        cpu_error = sum([abs(workload - target) for workload, target in zip(workload_cpu, self.cpu_target)])
        scanbytes_error = sum([abs(workload - target) for workload, target in zip(workload_scanbytes, self.scanbytes_target)])

        
        filter_error = sum([abs(workload - target) for workload, target in zip(workload_filter, self.filter_target)])
        sort_error = sum([abs(workload - target) for workload, target in zip(workload_sort, self.sort_target)])
        join_error = sum([abs(workload - target) for workload, target in zip(workload_join, self.join_target)])
        agg_error = sum([abs(workload - target) for workload, target in zip(workload_agg, self.agg_target)])
        
        global cpu_e, scan_e
        cpu_e = cpu_error
        scan_e = scanbytes_error
        
        global cpu_es, scan_es
        cpu_es = workload_cpu
        scan_es = workload_scanbytes
        
        global filter_es, sort_es, join_es, agg_es
        filter_es = [abs(workload - target) for workload, target in zip(workload_filter, self.filter_target)]
        sort_es = [abs(workload - target) for workload, target in zip(workload_sort, self.sort_target)]
        join_es = [abs(workload - target) for workload, target in zip(workload_join, self.join_target)]
        agg_es = [abs(workload - target) for workload, target in zip(workload_agg, self.agg_target)]
        
        return cpu_error + scanbytes_error

def delete_space(s):
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    s = s.replace("\r", "")
    s = s.replace(" ", "")
    s = s.replace("'", "")
    s = s.replace('"', "")
    s = s.replace(";", "")
    s = s.replace("\\", "")
    return s

def set_query_timestamp(config, workload, plans):

    sql_candidates = []
    for query_set, database in zip(config["query"], config["db"]):
        records = read_sql_records(query_set, database)
        for record in records:
            record["avg_scan_bytes"] = record["avg_scan_bytes"] / (1024 ** 3)
            if "@" not in record["query"]:
                record["query"] = record["query"] + "@" + database
            sql_candidates.append(record)
    
    for plan in plans:
        for query, count in plan["queries"].items():
            for candidate in sql_candidates:
                if delete_space(candidate["query"]) == delete_space(query):
                    plan["queries"][query] = {
                        "count": count,
                        "interval": config["interval"],
                        "avg_cpu_time": candidate["avg_cpu_time"],
                        "avg_scan_bytes": candidate["avg_scan_bytes"],
                        "avg_duration": candidate["avg_duration"],
                        "avg_lasted": int(candidate["avg_duration"] // config["interval"] + 1),
                        "filter": candidate["filter"],
                        "agg": candidate["agg"],
                        "sort": candidate["sort"],
                        "join": candidate["join"]
                    }
    cputime_interval = workload["cputime_interval"].to_list()
    cputime_interval = [json.loads(x) for x in cputime_interval]

    scanbytes_interval = workload["scanbytes_interval"].to_list()
    scanbytes_interval = [json.loads(x) for x in scanbytes_interval]
    
    filter_interval = workload["filter_interval"].to_list()
    filter_interval = [json.loads(x) for x in filter_interval]
    
    sort_interval = workload["sort_interval"].to_list()
    sort_interval = [json.loads(x) for x in sort_interval]
    
    join_interval = workload["join_interval"].to_list()
    join_interval = [json.loads(x) for x in join_interval]
    
    agg_interval = workload["agg_interval"].to_list()
    agg_interval = [json.loads(x) for x in agg_interval]
    
    for i in range(len(plans)):
        if len(plans[i]["queries"]) == 0:
            continue
        cpu_target = cputime_interval[i]
        scanbytes_target = scanbytes_interval[i]
        filter_target = filter_interval[i]
        sort_target = sort_interval[i]
        join_target = join_interval[i]
        agg_target = agg_interval[i]
        queries = []
        for query, info in plans[i]["queries"].items():
            for _ in range(int(info["count"])):
                queries.append({
                    "query": query,
                    "info": info
                })
        initial_solution = [0 for _ in range(len(queries))]
        problem = Problem(initial_solution, queries, cpu_target, scanbytes_target, filter_target, sort_target, join_target, agg_target)
        problem.set_schedule(problem.auto(minutes=1, steps=100))
        problem.copy_strategy = "slice"
        state, e = problem.anneal()
        
        print(cpu_es, scan_es)

        tmp = [[] for _ in range(len(cpu_target))]
        for j, query in enumerate(queries):
            tmp[state[j]].append(query["query"])

        plans[i]["queries"] = tmp

    return plans


def save_plan(config, results):
    """ Save the optimization plan to a JSON file. """
    workload = config["workload_name"]
    back = "+".join(sorted(config["query"]))
    plan_path = f"./output/sa_plan/{workload}/{back}-plan2.json"
    with open(plan_path, "w") as f:
        json.dump(results, f, indent=2)


def load_plan(config):
    """ Load the execution plan from a JSON file. """
    workload_name = config["workload_name"]
    back = "+".join(sorted(config["query"]))
    plan_path = f"./output/plan/{workload_name}/{back}-plan.json"    
    with open(plan_path, "r") as f:
        return json.load(f)


def load_workload(config):
    workload = pd.read_csv(config["workload_path"])
    return workload

def gen_ta(config):
    import time
    s = time.time()
    plans = load_plan(config)
    workload = load_workload(config)
    plans = set_query_timestamp(config, workload, plans)
    e = time.time()
    print("set_query_timestamp: ", e - s)
    save_plan(config, plans)
