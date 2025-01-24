import itertools
import json
import os
import random
import time
from datetime import datetime

from common.prometheus import prometheus_queries
# from common.record_time import get_time
from dotenv import load_dotenv

from .driver import BenchmarkDriver

import sys
sys.path.append("/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/metrics_witho")
from prometheus import prometheus_queries

def get_time():
    # get local time
    timestamp = time.time()
    # timestamp = time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime(timestamp))
                        
    return timestamp

def load_config():
    """ Load configuration from environment variables. """
    load_dotenv()
    return {
        "host": os.getenv("HOST"),
        "databend_port": os.getenv("DATABEND_PORT"),
        "prometheus_port": os.getenv("PROMETHEUS_PORT"),
        "plan": os.getenv("LEARN_PLAN"),
        "wait": int(os.getenv("WAIT_TIME", 0)),
        "seconds_in_time_slot": int(os.getenv("LEARN_SECONDS_IN_TIME_SLOTS", 0)),
        "workload_name": os.getenv("WORKLOAD_NAME", "default")
    }


def load_benchmark_configurations(plan, config):
    workload_name = config["workload_name"]
    plan_file = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/baseline_test/stitcher/output/{workload_name}/{plan}-plan.json"
    with open(plan_file, "r") as f:
        return json.load(f)

def collect_metrics(idx, start_time, end_time, start_cputime, end_cputime, start_scan, end_scan, wait_time):
    """ Collect and return metrics for a single execution. """
    return idx, end_cputime - start_cputime, end_scan - start_scan, end_time - start_time - wait_time

def execute_benchmarks(benchmark_configurations, config):
    """ Execute benchmark with provided configurations. """
    data = []
    for idx, configuration in enumerate(benchmark_configurations):
        drivers = _initialize_drivers(configuration, config)
        start_time, start_cputime, start_scan = _start_benchmark(drivers, config)

        time.sleep(config["seconds_in_time_slot"])
        _terminate_drivers(drivers)

        time.sleep(config["wait"])
        end_time, end_cputime, end_scan = _end_benchmark(drivers, config)
        
        cpu = []
        scan = []
        for t in range(int(start_time), int(start_time + 300 + config["interval"]), config["interval"]):
            cpu_time = prometheus_queries["cpu_new"](config["host"], config["prometheus_port"], t)
            scan_bytes = prometheus_queries["scan"](config["host"], config["prometheus_port"], t)
            cpu.append(cpu_time)
            scan.append(scan_bytes)

        cpu = [cpu[i] - cpu[i - 1] for i in range(1, len(cpu))]
        scan = [scan[i] - scan[i - 1] for i in range(1, len(scan))]
        scan = [s / (1024 ** 3) for s in scan]

        idx, cpu_time, scan_bytes, duration = collect_metrics(idx, start_time, end_time, start_cputime, end_cputime, start_scan, end_scan, config["wait"])
        data.append({
            "idx": idx,
            "cpu_time_sum": cpu_time,
            "scan_bytes_sum": scan_bytes,
            "cpu_time_interval": cpu,
            "scan_time_interval": scan,
            "duration": duration
            #,**operator_ratios
        })

        _output_results(config["plan"], data, config)


def _initialize_drivers(configuration, config):
    """ Initialize benchmark drivers based on the configuration. """
    drivers = []
    for benchmark in configuration:
        for _ in range(benchmark["terminal"]):
            drivers.append(BenchmarkDriver(config["host"], config["databend_port"], config["seconds_in_time_slot"], benchmark["benchmark"], benchmark["database"], benchmark["frequency"]))
    return drivers


def _start_benchmark(drivers, config):
    """ Start benchmark test and record start time and metrics. """
    start_time = get_time()
    print(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    start_cputime = prometheus_queries["cpu_new"](config["host"], 9090, start_time)
    start_scan = prometheus_queries["scan"](config["host"], 9090, start_time)
    for driver in drivers:
        driver.run()
    return start_time, start_cputime, start_scan


def _terminate_drivers(drivers):
    """ Terminate all running benchmark drivers. """
    for driver in drivers:
        driver.terminate()
    for driver in drivers:
        driver.wait()


def _end_benchmark(drivers, config):
    """ End benchmark test and record end time and metrics. """
    end_time = get_time()
    print(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    end_cputime = prometheus_queries["cpu_new"](config["host"], 9090, end_time)
    end_scan = prometheus_queries["scan"](config["host"], 9090, end_time)
    return end_time, end_cputime, end_scan

def _output_results(plan, data, config):
    """ Output the results to a file. """
    workload_name = config["workload_name"]
    record_file = f"/Users/zsy/Documents/codespace/python/FlexBench_original/simulator/rushrush/baseline_test/stitcher/output/{workload_name}/{plan}-results.json"
    with open(record_file, "w") as f:
        json.dump(data, f, indent=2)
    print("-" * os.get_terminal_size().columns)

def main():
    config = load_config()
    benchmark_configurations = load_benchmark_configurations(config["plan"], config)
    execute_benchmarks(benchmark_configurations, config)

def do_stitcher_replay(in_config):
    config = load_config()
    config["host"] = in_config["host"]
    config["databend_port"] = in_config["databend_port"]
    config["prometheus_port"] = in_config["prometheus_port"]
    config["seconds_in_time_slot"] = in_config["seconds_in_time_slot"]
    config["workload_name"] = in_config["workload_name"]
    config["interval"] = in_config["interval"]
    benchmark_configurations = load_benchmark_configurations(config["plan"], config)
    execute_benchmarks(benchmark_configurations, config)


if __name__ == "__main__":
    main()
