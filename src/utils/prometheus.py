from functools import partial

import requests


def query_prometheus(host, port, time, query):
    prometheus_url = f"http://{host}:{port}/api/v1/query"
    try:
        params = {"query": query, "time": time}
        response = requests.get(prometheus_url, params=params, timeout=10)  
        res = response.json()
        if res["status"] == "success":
            return float(res["data"]["result"][0]["value"][1])
        return 0.0
    except Exception as _:
        return 0.0


prometheus_queries = {
    # "cpu_new": partial(query_prometheus, query="sum(databend_process_cpu_seconds_total_total)"),
    "cpu_new": partial(query_prometheus, query="sum(max(databend_process_cpu_seconds_total_total) by (instance))"),
    # "cpu_new": partial(query_prometheus, query="sum(node_cpu_seconds_total{mode=~\"system|user\"})"),
    # "scan": partial(query_prometheus, query="sum(databend_query_scan_bytes_total)")
    "scan": partial(query_prometheus, query="sum(max(databend_query_scan_bytes_total) by (instance))")
}