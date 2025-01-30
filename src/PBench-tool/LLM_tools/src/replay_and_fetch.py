

def record_operator(host, databend_port, query,database):
    """ Record the operators used in the query. """
     # TODO: These are the keywords for the operators in the Databend execution plan. Update them as needed.
    dic = {
        "filter": ["Filter"],
        "join": ["HashJoin"],
        "agg": ["AggregateFinal", "AggregatePartial"],
        "sort": ["Sort"]
    }
    plan = execute_query(host, databend_port, query,database)
    operator_cnt = {
        "filter": 0,
        "join": 0,
        "agg": 0,
        "sort": 0
    }
    for i in range(len(plan)):
        tmp = '\n'.join([row[0] for row in plan[i][2]])
        for operator, keyword in dic.items():
            for k in keyword:
                operator_cnt[operator] += tmp.count(k)
    return operator_cnt


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


def replay_and_fetch(host, databend_port, prometheus_port, query, wait_time,database):
    repeat=3
    operator_cnt=record_operator(host, databend_port, query,database)
    cpu_sum=0
    scan_sum=0
    duration_sum=0
    for i in range(repeat):
        query, cpu_time, scan_bytes, duration = record_metrics(host, databend_port, prometheus_port, query, wait_time,database)
        cpu_sum+=cpu_time
        scan_sum+=scan_bytes
        duration_sum+=duration
    cpu_avg=cpu_sum/repeat
    scan_avg=scan_sum/repeat
    duration_avg=duration_sum/repeat
    return operator_cnt,cpu_avg,scan_avg,duration_avg