<p align="center">
    <h3 align="center">FlexBench</h3>
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

Python 3.10 is required to run FlexBench. To set up the environment, follow the steps below:

1. Install Python 3.10

    ```
    sudo apt-get install python3.10
    ```

2. Install required packages

    ```
    pip install -r requirements.txt
    ```

# Workload

[Snowset](https://github.com/resource-disaggregation/snowset) contains several statistics (timing, I/O, resource usage, etc..) pertaining to ~70 million queries from all customers that ran on [Snowflake](https://www.snowflake.com/) over a 14 day period from Feb 21st 2018 to March 7th 2018. FlexBench uses the statistics in Snowset to synthesize database workloads.

# Usage

FlexBench can synthesize database workloads using different methods. The following sections describe how to use each method.

## Configuration

Conguration can be set in `simulator/.env`.

- `WORKLOAD_PATH` is the path to the workload file.
- `HOST` is the ip address of the Databend server.
- `DATABEND_PORT` is the port of the Databend.
- `PROMETHEUS_PORT` is the port of the Prometheus.
- `WAIT_TIME` is the time(in second) after playing workload and before collecting statistics.
- `TIME_SLOTS` is the number of time slots in the database workload.

## Stitcher

To synthesize database workloads by Stitcher, follow the steps below:

1. Collect the statistics of benchmark pieces

    ```
    PYTHONPATH=[path to FlexBench]/simulator python simulator/learning/collect.py 
    ```

2. Synthesize database workloads

    ```
    PYTHONPATH=[path to FlexBench]/simulator python simulator/learning/stitcher.py
    ```

3. Play the synthesized database workloads

    ```
    PYTHONPATH=[path to FlexBench]/simulator python simulator/learning/replay.py
    ```

Parameters of Stitcher can also be set in `simulator/.env`.

- `STITCHER_SEED` is the random seed for Stitcher.
- `STITCHER_ITER` is the number of iterations for Bayesian Optimization in Stitcher.
- `LEARN_SECONDS_IN_TIME_SLOTS` is the seconds in each time slot.
- `LEARN_BENCHMARK` is the benchmark pieces used in Stitcher.

    - e.g. `LEARN_BENCHMARK=TPCH,ycsb` means Stitcher uses `TPCH` and `ycsb`.

- `LEARN_[benchmark]_DATABASE` is the database used for the given benchmark piece.

    - e.g. `LEARN_TPCH_DATABASE=tpch5g` means the database corresponding to `TPCH` is `tpch5g`.

- `LEARN_[database]_MIN_TERMINAL` is the minimum terminal for the given benchmark piece.
- `LEARN_[database]_MAX_TERMINAL` is the maximum terminal for the given benchmark piece.
- `LEARN_[database]_TERMINAL_INTERVAL` is sampling interval of terminals for the given benchmark piece.
- `LEARN_[database]_MIN_FREQUENCY` is the minmum upper limit of submitted queries from one client per minure
- `LEARN_[database]_MAX_FREQUENCY` is the maximum upper limit of submitted queries from one client per minure
- `LEARN_[database]_FREQUENCY_INTERVAL` is sampling interval of frequency for the given benchmark piece.

## Conditional Variational Autoencoder

Same as Stitcher, as they both synthesize database workloads by borrowing pieces of existing benchmarks.

## Integer Linear Programming

To synthesize database workloads by Integer Linear Programming, follow the steps below:

1. Collect the statistics of queries

    ```
    PYTHONPATH=[path to FlexBench]/simulator python simulator/linear/collect.py 
    ```

2. Synthesize database workloads

    ```
    PYTHONPATH=[path to FlexBench]/simulator python simulator/linear/linear.py
    ```

3. Play the synthesized database workloads

    ```
    PYTHONPATH=[path to FlexBench]/simulator python simulator/linear/replay.py
    ```

Parameters of Integer Linear Programming can also be set in `simulator/.env`.

- `LP_QUERY_SET` is the query set used for Integer Linear Programming.
- `LP_DATABASE` is the database used for Integer Linear Programming.
- `LP_COUNT_LIMIT` is the upper limit of queries in a time slot.
- `LP_TIME_LIMIT` currently not used.
