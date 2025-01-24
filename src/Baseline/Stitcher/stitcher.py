import json
import os
from functools import partial

import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from .util import load_data, seed_everything, load_data_from_single_benchmark

def train_model(df, metrics_columns):
    """ Train a linear regression model on the given DataFrame. """
    benchmark_pieces_columns = sorted(
        [col for col in df.columns if col not in metrics_columns],
        key=lambda x: (x.split('_')[0], x.split('_')[1])
    )

    y_train = df[metrics_columns].values
    x_train = df[benchmark_pieces_columns].values

    model = MultiOutputRegressor(LinearRegression(fit_intercept=False))
    model.fit(x_train, y_train)

    return model, benchmark_pieces_columns


def target_function(model, y, **params):
    """ Objective function for Bayesian Optimization. """
    x = []
    for i in range(len(params.keys())):
        x.append(params[f"x{i}"])
    predicted_output = model.predict([x])
    mspe = (1/len(y)) * sum(((predicted_output[0] - y)**2 / y**2) * 100)
    return -mspe

def load_workload(config):
    """ Load workload data from a CSV file and preprocess it. """
    workload = pd.read_csv(config["workload_path"])
    # workload = workload.iloc[:config["slots"], :]
    return workload


def optimize_and_record_results(model, benchmarks, y_test, benchmark_pieces_columns, config):
    """ Optimize model parameters using Bayesian Optimization and record results. """
    results = []
    for i in range(y_test.shape[0]):
        y = y_test[i]
        # y is a vector and if any of the values is 0, results.append[0]
        if 0 in y:
            result = []
            for benchmark, database in benchmarks:
                result.append({
                    "benchmark": benchmark,
                    "database": database,
                    "frequency": 0,
                    "terminal": 0
                })
            results.append(result)
            continue
        pbounds = {}
        for j in range(int(model.n_features_in_)):
            pbounds[f"x{j}"] = (0, 60)
            
        optimizer = BayesianOptimization(
            f=partial(target_function, model=model, y=y),
            pbounds=pbounds,
            allow_duplicate_points=True,
            random_state=config["seed"]
        )
        optimizer.maximize(n_iter=config["iter"])
        configuration = {}
        test_x = []
        for j, column in enumerate(benchmark_pieces_columns):
            configuration[column] = optimizer.max["params"][f"x{j}"]
            test_x.append(configuration[column])
        predicted = model.predict([test_x])
        result = []
        for benchmark, database in benchmarks:
            aux = configuration[f"{database}_auxiliarynew"]     
            if aux < 0.99:
                terminal = 0
                frequency = 0
            else:
                terminal=1
                frequency = int(aux/terminal)
                while frequency > 270:
                    terminal +=1
                    frequency = int(aux/terminal)
            result.append({
                "benchmark": benchmark,
                "database": database,
                "frequency": frequency,
                "terminal": terminal,
            })
        print(test_x)
        print(predicted)
        results.append(result)

        workload_name = config["workload_name"]
        record_file = f"./output/{workload_name}/stitcher-plan.json"
        with open(record_file, "w") as f:
            json.dump(results, f, indent=2)

def gen_stitcher_plan(config):
    seed_everything(config["seed"])
    profile, benchmarks = load_data()
    model, benchmark_pieces_columns = train_model(profile, ["cputime", "scanbytes"])

    workload = load_workload(config)
    y_test = workload[["cputime_sum", "scanbytes_sum"]].values
    
    if "redset" in config["workload_name"]:
        for i in range(len(y_test)):
            y_test[i][1] = y_test[i][1] / 1024

    benchmark_pieces_columns = [columns for columns in          benchmark_pieces_columns if not columns.endswith('_auxiliary')]
    optimize_and_record_results(model, benchmarks, y_test, benchmark_pieces_columns, config)

def gen_stitcher(config, benchmark_name):
    seed_everything(config["seed"])
    profile, benchmarks = load_data_from_single_benchmark(benchmark_name)
    model, benchmark_pieces_columns = train_model(profile, ["cputime", "scanbytes"])

    workload = load_workload(config)
    y_test = workload[["cputime_sum", "scanbytes_sum"]].values

    benchmark_pieces_columns = [columns for columns in          benchmark_pieces_columns if not columns.endswith('_auxiliary')]
    optimize_and_record_results(model, benchmarks, y_test, benchmark_pieces_columns, config)
    
    return model
    
def find_best_plan(config, benchmark_names):
    for name in benchmark_names:
        model = gen_stitcher(config, name)
        