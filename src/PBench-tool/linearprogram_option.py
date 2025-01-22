import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pulp
from LLM_new.src.llm_gen import generate_query
import time

def get_time():
    # get local time
    timestamp = time.time()
    # timestamp = time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime(timestamp))
                        
    return timestamp

def read_sql_records(query_set, database):
    """ Read SQL records from a JSON file. """
    record_file = os.path.join(f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/metrics_witho/output/{query_set}-{database}-sql-metrics.json")
    with open(record_file, "r") as f:
        return json.load(f)


def solve_integer_linear_programming_with_normalization(config, items, target_cpu_time, target_scan_bytes, target_duration, time_limit, target_filter=None, target_join=None, target_agg=None, target_sort=None, count_limit=None,real_count=None):
    # for i in range(len(items)):
    #     if items[i]["query"].startswith("EXPLAIN ANALYZE SELECT 1"):
    #         #在这个列表中删除这一项
    #         print(items[i]["query"])
    #         items.pop(i)
    """ Solve the integer linear programming problem with normalization. """
    prob = pulp.LpProblem("MinimizeDifference", pulp.LpMinimize)
    item_vars = [pulp.LpVariable(f"n_{i}", lowBound=0, cat='Integer') for i in range(len(items))]
    cpu_var = pulp.LpVariable("cpu", lowBound=0)
    scan_var = pulp.LpVariable("scan", lowBound=0)
    if config["use_duration"]:
        duration_var = pulp.LpVariable("duration", lowBound=0)
        prob += duration_var * config["duration_scale"]
    if config["use_operator"]:
        filter_var = pulp.LpVariable("filter", lowBound=0)
        join_var = pulp.LpVariable("join", lowBound=0)
        agg_var = pulp.LpVariable("agg", lowBound=0)
        sort_var = pulp.LpVariable("sort", lowBound=0)
        prob += (filter_var + join_var + agg_var + sort_var) * config["op_scale"] + (cpu_var + scan_var)
    else:
        prob += (cpu_var + scan_var)

    # Handle the special case
    for i in range(len(items)):
        if items[i]["query"].startswith("EXPLAIN ANALYZE SELECT 1"):
            items[i]["factor"] = 1
        else:
            items[i]["factor"] = 1

    # Performance Metrics
    # prob += ((sum([item["factor"] * item["avg_cpu_time"] * item_vars[i] for i, item in enumerate(items)]) - target_cpu_time) / target_cpu_time) <= cpu_var
    # prob += ((target_cpu_time - sum([item["factor"] * item["avg_cpu_time"] * item_vars[i] for i, item in enumerate(items)])) / target_cpu_time) <= cpu_var
    prob += sum([item["factor"] * item["avg_cpu_time"] * item_vars[i] for i, item in enumerate(items)]) - target_cpu_time <= cpu_var
    prob += target_cpu_time - sum([item["factor"] * item["avg_cpu_time"] * item_vars[i] for i, item in enumerate(items)]) <= cpu_var

    # prob += ((sum([item["factor"] * item["avg_scan_bytes"] * item_vars[i] for i, item in enumerate(items)]) - target_scan_bytes) / target_scan_bytes) <= scan_var
    # prob += ((target_scan_bytes - sum([item["factor"] * item["avg_scan_bytes"] * item_vars[i] for i, item in enumerate(items)])) / target_scan_bytes) <= scan_var
    prob += sum([item["factor"] * item["avg_scan_bytes"] * item_vars[i] for i, item in enumerate(items)]) - target_scan_bytes <= scan_var
    prob += target_scan_bytes - sum([item["factor"] * item["avg_scan_bytes"] * item_vars[i] for i, item in enumerate(items)]) <= scan_var
    real_count = max(real_count, 1)

    # Query Plan Operator Ratios
    if config["use_operator"]:
        for attr in ["filter", "join", "agg", "sort"]:
            factor_sum = sum(item["factor"] * item[attr] * item_vars[i] for i, item in enumerate(items))
            target_diff = locals()["target_" + attr] - factor_sum / real_count
            prob += (target_diff) <= locals()[attr + "_var"]
            prob += (-target_diff) <= locals()[attr + "_var"]
        
    if config["use_duration"]:
        factor_sum = sum(item["factor"] * item["avg_duration"] * item_vars[i] for i, item in enumerate(items))
        target_diff = target_duration - factor_sum / real_count
        prob += (target_diff) <= target_duration
        prob += (-target_diff) <= target_duration

    # prob += sum([item["factor"] * item_vars[i] for i, item in enumerate(items)]) >= 0.6 * count_limit
    prob += sum([item["factor"] * item_vars[i] for i, item in enumerate(items)]) <= count_limit
    
    prob += sum([item["factor"] * item["avg_duration"] * item_vars[i] for i, item in enumerate(items)]) <= time_limit
    for i in range(len(items)):
        prob += item_vars[i] <= 10
    
    prob.writeLP("MinimizeDifference3.lp")
    prob.solve(solver=pulp.PULP_CBC_CMD(timeLimit=10, msg=False))
    # prob.solve(solver=pulp.PULP_CBC_CMD(msg=False))
    print(pulp.LpStatus[prob.status])
    item_counts = [item_vars[i].varValue * items[i]["factor"] for i in range(len(items))]
    min_diff = pulp.value(prob.objective)

    return min_diff, item_counts


def solve_integer_linear_programming_cycle(config, items, target_cpu_time, target_scan_bytes, target_duration, time_limit, target_filter=None, target_join=None, target_agg=None, target_sort=None, count_limit=None,init_count=None):
    min_diff, solution = solve_integer_linear_programming_with_normalization(config, items, target_cpu_time, target_scan_bytes, target_duration, time_limit, target_filter, target_join, target_agg, target_sort, count_limit,real_count=init_count)
    sql_count = dict(zip([json.dumps(candidate["query"]) for candidate in items], solution))
    total_count = sum(sql_count.values())
    max_try = 4
    print(f"Total Count: {total_count}, Init Count: {init_count}")
    while (abs(total_count-init_count) >= 0.13 * init_count) and max_try > 0:
        max_try -= 1
        init_count = total_count
        min_diff, solution = solve_integer_linear_programming_with_normalization(config, items, target_cpu_time, target_scan_bytes, target_duration, time_limit, target_filter, target_join, target_agg, target_sort, count_limit,real_count=init_count)
        sql_count = dict(zip([json.dumps(candidate["query"]) for candidate in items], solution))
        total_count = sum(sql_count.values())
        print(f"Total Count: {total_count}, Init Count: {init_count}")
    return min_diff, solution

def create_sql_candidates_pool(config):
    sql_candidates = []
    for query_set, database in zip(config["query"], config["db"]):
        records = read_sql_records(query_set, database)
        for record in records:
            # if record["avg_scan_bytes"]==0:
            #     continue
            # if record["avg_cpu_time"]<0.1:
            #     continue
            # if record["avg_cpu_time"] / (record["avg_scan_bytes"] / (1024**3)) > 100:
            #     continue
            record["avg_scan_bytes"] = record["avg_scan_bytes"] / (1024 ** 3)
            if "@" not in record["query"]:
                record["query"] += "@" + database
            sql_candidates.append(record)
    return sql_candidates 

    
def generate_workload(config,is_test):
    """ Generate the workload data and solve optimization problems. """
    workload = pd.read_csv(config["workload_path"])
    if config["use_operator"]:
        workload_time_slots = list(zip(workload['cputime_sum'], 
                                        workload['scanbytes_sum'], 
                                        workload['avg_durationtime'],
                                        workload['filter'], 
                                        workload['join'], 
                                        workload['agg'], 
                                        workload['sort']
                                    ))
    else:
        workload_time_slots = list(zip(workload['cputime_sum'], workload['scanbytes_sum']))
    
    results, effects = [], []
    import time
    total_time = 0
    if config["use_operator"]:
        for cpu_time, scan_bytes, duration, filter, join, agg, sort in workload_time_slots[:]:
            sql_candidates=create_sql_candidates_pool(config)
            if cpu_time == 0 and scan_bytes == 0:
                results.append({
                    "queries": {},
                    "operator_ratios": {
                        "filter": 0,
                        "join": 0,
                        "agg": 0,
                        "sort": 0
                    }
                })
                continue
            begin = get_time()
            min_diff, solution = solve_integer_linear_programming_cycle(
                config, sql_candidates, cpu_time, scan_bytes, duration, time_limit=config["time_limit"], target_filter=filter, target_join=join, target_agg=agg, target_sort=sort, count_limit=config["count_limit"],init_count=config["initial_count"]
            )
            end = get_time()
            total_time += end - begin

            sql_count = dict(zip([json.dumps(candidate["query"]) for candidate in sql_candidates], solution)) 
            # sql_count = dict(zip([candidate["query"] for candidate in sql_candidates], solution)) 
            sqls = {sql: count for sql, count in sql_count.items() if count > 0}
            
            # Performance Metrics
            cpu_sum = sum([candidate["avg_cpu_time"] * sql_count for candidate, sql_count in zip(sql_candidates, solution)])
            scan_sum = sum([candidate["avg_scan_bytes"] * sql_count for candidate, sql_count in zip(sql_candidates, solution)])

            # Query Plan Operator Ratios
            duration_sum = sum([candidate["avg_duration"] * sql_count for candidate, sql_count in zip(sql_candidates, solution)])
            filter_count = sum([filter_per_sql * sql_count for filter_per_sql, sql_count in zip([candidate["filter"] for candidate in sql_candidates], solution)])
            join_count = sum([join_per_sql * sql_count for join_per_sql, sql_count in zip([candidate["join"] for candidate in sql_candidates], solution)])
            agg_count = sum([agg_per_sql * sql_count for agg_per_sql, sql_count in zip([candidate["agg"] for candidate in sql_candidates], solution)])
            sort_count = sum([sort_per_sql * sql_count for sort_per_sql, sql_count in zip([candidate["sort"] for candidate in sql_candidates], solution)])
            sql_count = sum(sql_count.values())
            if sql_count == 0:
                avg_duration = 0
                filter_ratio = 0
                join_ratio = 0
                agg_ratio = 0
                sort_ratio = 0
            else:
                avg_duration = duration_sum / sql_count
                filter_ratio = filter_count / sql_count
                join_ratio = join_count / sql_count
                agg_ratio = agg_count / sql_count
                sort_ratio = sort_count / sql_count

            print(f"Target  : CPU: {cpu_time:.2f}, Scan: {scan_bytes:.2f}, Duration: {duration:.2f}, Filter: {filter:.2f}, Join: {join:.2f}, Agg: {agg:.2f}, Sort: {sort:.2f}")
            print(f"Generate: CPU: {cpu_sum:.2f}, Scan: {scan_sum:.2f}, Duration: {avg_duration:.2f}, Filter: {filter_ratio:.2f}, Join: {join_ratio:.2f}, Agg: {agg_ratio:.2f}, Sort: {sort_ratio:.2f}")

            # compute MAE between target and generated
            cpu_diff = abs(cpu_time - cpu_sum) / (cpu_time + 1)
            scan_diff = abs(scan_bytes - scan_sum) / (scan_bytes + 1)
            duration_diff = abs(duration - avg_duration) / (duration + 1)
            Filter_diff = abs(filter - filter_ratio)
            Join_diff = abs(join - join_ratio)
            Agg_diff = abs(agg - agg_ratio)
            Sort_diff = abs(sort - sort_ratio)
            if config["use_duration"]:
                total_MAPE = (cpu_diff + scan_diff + duration_diff + Filter_diff + Join_diff + Agg_diff + Sort_diff) / 7
            else:
                total_MAPE = (cpu_diff + scan_diff + Filter_diff + Join_diff + Agg_diff + Sort_diff) / 6
            print(f"MAPE     : CPU: {cpu_diff:.2f}, Scan: {scan_diff:.2f}, Duration: {duration_diff:.2f}, Filter: {Filter_diff:.2f}, Join:{Join_diff:.2f},Agg:{Agg_diff:.2f},Sort:{Sort_diff:.2f},total_MAPE:{total_MAPE:.2f}")
            print("-" * os.get_terminal_size().columns)
            times=0
            # llm code need to be updated
            while times > 0 and total_MAPE > 0.1:
                times-=1
                generate_query(config, cpu_time, scan_bytes, filter, join, agg, sort)
                sql_candidates=create_sql_candidates_pool(config)
                begin = get_time()
                min_diff, solution = solve_integer_linear_programming_cycle(
                    config, sql_candidates, cpu_time, scan_bytes, duration, time_limit=config["time_limit"], target_filter=filter, target_join=join, target_agg=agg, target_sort=sort, count_limit=config["count_limit"],init_count=config["initial_count"]
                )
                end = get_time()
                total_time += end - begin
                sql_count = dict(zip([json.dumps(candidate["query"]) for candidate in sql_candidates], solution)) 
                # sql_count = dict(zip([candidate["query"] for candidate in sql_candidates], solution)) 
                sqls = {sql: count for sql, count in sql_count.items() if count > 0}
                # Performance Metrics
                cpu_sum = sum([candidate["avg_cpu_time"] * sql_count for candidate, sql_count in zip(sql_candidates, solution)])
                scan_sum = sum([candidate["avg_scan_bytes"] * sql_count for candidate, sql_count in zip(sql_candidates, solution)])
                # Query Plan Operator Ratios
                filter_count = sum([filter_per_sql * sql_count for filter_per_sql, sql_count in zip([candidate["filter"] for candidate in sql_candidates], solution)])
                join_count = sum([join_per_sql * sql_count for join_per_sql, sql_count in zip([candidate["join"] for candidate in sql_candidates], solution)])
                agg_count = sum([agg_per_sql * sql_count for agg_per_sql, sql_count in zip([candidate["agg"] for candidate in sql_candidates], solution)])
                sort_count = sum([sort_per_sql * sql_count for sort_per_sql, sql_count in zip([candidate["sort"] for candidate in sql_candidates], solution)])
                sql_count = sum(sql_count.values())
                if sql_count == 0:
                    filter_ratio = 0
                    join_ratio = 0
                    agg_ratio = 0
                    sort_ratio = 0
                else:
                    filter_ratio = filter_count / sql_count
                    join_ratio = join_count / sql_count
                    agg_ratio = agg_count / sql_count
                    sort_ratio = sort_count / sql_count
                print(f"Target  : CPU: {cpu_time:.2f}, Scan: {scan_bytes:.2f}, Filter: {filter:.2f}, Join: {join:.2f}, Agg: {agg:.2f}, Sort: {sort:.2f}")
                print(f"Generate: CPU: {cpu_sum:.2f}, Scan: {scan_sum:.2f}, Filter: {filter_ratio:.2f}, Join: {join_ratio:.2f}, Agg: {agg_ratio:.2f}, Sort: {sort_ratio:.2f}")
                # compute MAE between target and generated
                cpu_diff = abs(cpu_time - cpu_sum)/(cpu_time+1)
                scan_diff = abs(scan_bytes - scan_sum)/(scan_bytes+1)
                Filter_diff= abs(filter-filter_ratio)
                Join_diff=abs(join-join_ratio)
                Agg_diff=abs(agg-agg_ratio)
                Sort_diff=abs(sort-sort_ratio)
                total_MAPE=(cpu_diff+scan_diff+Filter_diff+Join_diff+Agg_diff*100+Sort_diff)/6
                print(f"MAPE     : CPU: {cpu_diff:.2f}, Scan: {scan_diff:.2f}, Filter: {Filter_diff:.2f}, Join:{Join_diff:.2f},Agg:{Agg_diff:.2f},Sort:{Sort_diff:.2f},total_MAPE:{total_MAPE:.2f}")
                print("-" * os.get_terminal_size().columns)
            results.append({
                "queries": sqls,
                "operator_ratios": {
                    "filter": filter_ratio,
                    "join": join_ratio,
                    "agg": agg_ratio,
                    "sort": sort_ratio
                }
            })
            effects.append(min_diff * 100)
            if is_test:
                break
    else:
        for cpu_time, scan_bytes in workload_time_slots[:]:
            begin = get_time()
            min_diff, solution = solve_integer_linear_programming_with_normalization(
                config, sql_candidates, cpu_time, scan_bytes, time_limit=config["time_limit"], count_limit=config["count_limit"]
            )
            end = get_time()
            total_time += end - begin

            sql_count = dict(zip([json.dumps(candidate["query"]) for candidate in sql_candidates], solution)) 
            # sql_count = dict(zip([candidate["query"] for candidate in sql_candidates], solution)) 
            sqls = {sql: count for sql, count in sql_count.items() if count > 0}
            
            # Performance Metrics
            cpu_sum = sum([candidate["avg_cpu_time"] * sql_count for candidate, sql_count in zip(sql_candidates, solution)])
            scan_sum = sum([candidate["avg_scan_bytes"] * sql_count for candidate, sql_count in zip(sql_candidates, solution)])

            print(f"Target  : CPU: {cpu_time:.2f}, Scan: {scan_bytes:.2f}")
            print(f"Generate: CPU: {cpu_sum:.2f}, Scan: {scan_sum:.2f}")

            # compute MAE between target and generated
            cpu_diff = abs(cpu_time - cpu_sum)/(cpu_time+1)
            scan_diff = abs(scan_bytes - scan_sum)/(scan_bytes+1)
            total_MAPE = (cpu_diff+scan_diff)/2
            print(f"MAE     : CPU: {cpu_diff:.2f}, Scan: {scan_diff:.2f}, Total MAPE: {total_MAPE:.2f}")
            print("-" * os.get_terminal_size().columns)

            results.append({
                "queries": sqls,
            })
            effects.append(min_diff * 100)

    print(f"Total Time: {total_time:.2f}")
    return results, np.mean(effects)


def save_plan(config, results):
    """ Save the optimization plan to a JSON file. """
    workload_name = config["workload_name"]
    back = "+".join(sorted(config["query"]))
    plan_path = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/output/plan/{workload_name}/{back}-plan.json"
    with open(plan_path, "w") as f:
        json.dump(results, f, indent=2)
    # reopen the file and delete the \\" in the file
    with open(plan_path, "r") as f:
        lines = f.readlines()
    with open(plan_path, "w") as f:
        for line in lines:
            f.write(line.replace('\\"', ''))

def ILP_work(config):
    s = time.time()
    results, avg_effect = generate_workload(config,is_test=False)
    e = time.time()
    print(f"Time: {e-s:.2f}")
    print(f"Average Effect: {avg_effect:.2f}%")
    save_plan(config, results)
def test_ILP(config):
    results, avg_effect = generate_workload(config,is_test=True)
    print(f"Average Effect: {avg_effect:.2f}%")
    save_plan(config, results)
    
    
