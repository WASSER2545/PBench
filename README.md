<p align="center">
    <h3 align="center">PBench</h3>
    <p align="center">A database workload synthesizer</p>
    <p align="center">
        <a href="#directory">Directory</a> •
        <a href="#environment">Environment</a> •
        <a href="#workload">Workload</a> •
        <a href="#usage">Usage</a>
    </p>
</p>

# Directory

```
> tree -L 2
..
├── README.md
├── requirements.txt
└── src
    ├── Baseline
    │   ├── CAB
    │   │   ├── cab_replay.py
    │   │   ├── gen.py
    │   │   ├── gen_redset.py
    │   │   ├── output
    │   │   │   ├── workload1h-5m-30s_1
    │   │   │   ├── workload1h-5m-30s_2
    │   │   │   ├── workload1h-5m-30s_3
    │   │   │   ├── workload1h-5m-30s_4
    │   │   │   └── workload1h-5m-30s_5
    │   │   └── prometheus.py
    │   ├── Stitcher
    │   │   ├── benchmark
    │   │   │   ├── TPCH.sql
    │   │   │   ├── imdb.sql
    │   │   │   ├── tpcds_all.sql
    │   │   │   ├── ycsb.py
    │   │   │   └── ycsb.sql
    │   │   ├── output
    │   │   │   ├── workload1h-5m-30s_1
    │   │   │   ├── workload1h-5m-30s_2
    │   │   │   ├── workload1h-5m-30s_3
    │   │   │   ├── workload1h-5m-30s_4
    │   │   │   ├── workload1h-5m-30s_5
    │   │   ├── profile.json
    │   │   ├── prometheus.py
    │   │   ├── replay.py
    │   │   ├── stitcher.py
    │   │   └── util.py
    │   └── do_baseline.py
    ├── PBench-tool
    │   ├── LLM_new
    │   │   ├── input
    │   │   │   ├── hints.csv
    │   │   │   ├── keys.txt
    │   │   │   └── table_schema
    │   │   │       ├── table_meta.json
    │   │   │       └── tpch1g.sql
    │   │   ├── output
    │   │   └── src
    │   │       ├── a.ipynb
    │   │       ├── inteaction.py
    │   │       ├── llm_gen.py
    │   │       ├── llm_gen_redset.py
    │   │       ├── llmapi.py
    │   │       ├── replay_and_fetch.py
    │   │       ├── retrieve_examples.py
    │   │       └── retrieve_meta.py
    │   ├── configs
    │   │   ├── workload1h-5m-30s_1
    │   │   ├── workload1h-5m-30s_2
    │   │   ├── workload1h-5m-30s_3
    │   │   ├── workload1h-5m-30s_4
    │   │   └── workload1h-5m-30s_5
    │   ├── do_sa.py
    │   ├── linearprogram_option.py
    │   ├── output
    │   ├── prometheus.py
    │   ├── random_send.py
    │   ├── replay_interval.py
    │   ├── replay_sa.py
    │   └── simulatedannealing.py
    └── Workloads
```

# Environment

Python 3.10 is required to run PBench. To set up the environment, follow the steps below:

1. Install Python 3.10

    ```
    sudo apt-get install python3.10
    ```

2. Install required packages

    ```
    pip install -r requirements.txt
    ```

# Workload

[Snowset](https://github.com/resource-disaggregation/snowset) contains several statistics (timing, I/O, resource usage, etc..) pertaining to ~70 million queries from all customers that ran on [Snowflake](https://www.snowflake.com/) over a 14 day period from Feb 21st 2018 to March 7th 2018. PBench uses the statistics in Snowset to synthesize database workloads.

# Usage

PBench can synthesize database workloads using different methods. The following sections describe how to use each method.

## Configuration

Conguration can be set in `PBench-tool/config/*.yml`.

This document provides a brief overview of the configuration parameters specified in the YAML file. The setup is designed to generate and execute workloads efficiently.

- **Workload Path**: `../../Workloads/Snowset/workload1h-5m-30s_1.csv`  
  This file specifies the original workload file.

- **Workload Name**: `workload1h-5m-30s_1`  
  A unique identifier for the workload, facilitating easy referencing and logging.

- **Count Limit**: `1000`  
  Sets the maximum number of operations or queries to be executed during the workload.

- **Time Limit**: `270` seconds  
  Defines the total duration within the time-window which all operations must be completed.

- **Use Operator**: `1`  
  Indicates whether the operation involves using operator target (value 1 suggests usage).

- **Interval**: `30` seconds  
  Determines the interval between operations or queries execution cycles.

- **Query Types**: `[TPCH, TPCH, TPCH, TPCH, tpcds_all, tpcds_all, imdb, llm]`  
  Lists the types of queries included in the workload, covering different datasets and benchmarks.

- **Database Names**: `[tpch500m, tpch1g, tpch5g, tpch9g, tpcds1g, tpcds2g, imdb, llm]`  
  Corresponds to the databases against which the queries will be executed, each representing different sizes or datasets.

- **Operator Scale**: `100`  
  Scaling factors affecting the operation's intensity or frequency.

- **Initial Count**: `10`  
  Specifies the initial target number of queries in time-window.

## Baseline experiments

The baseline tools include two widely known workload synthesizer: CAB and Stitcher. We provide our implementation and startup code in `Baseline/do_baseline.py`.

## PBench

To synthesize database workloads by PBench, follow the steps below:

1. Collect the statistics of queries

    ```
    PYTHONPATH=[path to PBench]/simulator python simulator/linear/collect.py 
    ```

2. Synthesize database workloads

    ```
    PYTHONPATH=[path to PBench]/simulator python simulator/linear/linear.py
    ```

3. Play the synthesized database workloads

    ```
    PYTHONPATH=[path to PBench]/simulator python simulator/linear/replay.py
    ```

Parameters of Integer Linear Programming can also be set in `simulator/.env`.

- `LP_QUERY_SET` is the query set used for Integer Linear Programming.
- `LP_DATABASE` is the database used for Integer Linear Programming.
- `LP_COUNT_LIMIT` is the upper limit of queries in a time slot.
- `LP_TIME_LIMIT` currently not used.
