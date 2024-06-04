import os
import re
import json
import pandas as pd
import datasets
import numpy as np
from main import configure, run


def show_log(output_dir, subset="eval_loss"):
    with open(os.path.join(output_dir, "trainer_state.json"), "r") as f:
        log_history = json.load(f)["log_history"]
    df_log = pd.DataFrame(log_history).dropna(subset=subset).dropna(how='all', axis=1)
    return df_log


def read_result(output_dir_format, dataset_names, key_column, init_dir, prefix="eval_loss_"):
    df_results = []
    for dataset_name in dataset_names:
        output_dir = output_dir_format.format(dataset_name)
        df_log = show_log(output_dir)
        df_log["dataset_name"] = dataset_name
        df_log = df_log.set_index("dataset_name")
        df_results.append(df_log)
    df_results = pd.concat(df_results)

    if prefix == "eval_loss_":
        df_results = df_results[[column for column in df_results.columns if column.startswith(prefix) or column == key_column]]
        df_results = df_results.rename(columns=lambda column: column.replace(prefix, ""))
    else:
        df_results = df_results[[prefix, key_column]]
    df_results = df_results.sort_index(axis=1)
    dict_results = {key: df_results[df_results[key_column] == key].drop(columns=[key_column]) for key in df_results[key_column].unique()}
    
    with open(os.path.join(init_dir, f"eval_results.json"), "r") as f:
        init_results = json.load(f)
    series_init = pd.Series(init_results)
    series_init = series_init.rename(index=lambda index: index.replace(prefix, ""))
    
    dict_result = list(dict_results.values())[0]
    series_init = series_init.loc[dict_result.columns]
    df_init = pd.DataFrame.from_dict({index: series_init for index in dict_result.index}, orient="index")
    df_init.index.name = dict_result.index.name
    
    dict_results[0] = df_init    
    dict_results = {key: df_result-df_init for key, df_result in dict_results.items()}
        
    df_results = []
    for key, df_result in dict_results.items():
        df_result[key_column] = key
        df_results.append(df_result)

    df_result = pd.concat(df_results)
    return df_result


def read_result_reverse(output_dir_format, dataset_names, key_column, init_dir, prefix="eval_loss_"):
    df_results = []
    for dataset_name in dataset_names:
        output_dir = output_dir_format.format(dataset_name)
        df_log = show_log(output_dir)
        df_log["dataset_name"] = dataset_name
        df_log = df_log.set_index("dataset_name")
        df_results.append(df_log)
    df_results = pd.concat(df_results)

    if prefix == "eval_loss_":
        df_results = df_results[[column for column in df_results.columns if column.startswith(prefix) or column == key_column]]
        df_results = df_results.rename(columns=lambda column: column.replace(prefix, ""))
    else:
        df_results = df_results[[prefix, key_column]]
    df_results = df_results.sort_index(axis=1)
    dict_results = {key: df_results[df_results[key_column] == key].drop(columns=[key_column]).T for key in df_results[key_column].unique()}
    
    with open(os.path.join(init_dir, f"eval_results.json"), "r") as f:
        init_results = json.load(f)
    series_init = pd.Series(init_results)
    series_init = series_init.loc[[index for index in series_init.index if index.startswith(prefix)]]
    series_init = series_init.rename(index=lambda index: index.replace(prefix, ""))
    
    dict_result = list(dict_results.values())[0]
    series_init = series_init.loc[dict_result.index]
    df_init = pd.DataFrame({column: series_init for column in dict_result.columns})
    df_init.index.name = dict_result.index.name
    
    dict_results[0] = df_init    
    dict_results = {key: df_result-df_init for key, df_result in dict_results.items()}
    
    df_results = []
    for key, df_result in dict_results.items():
        df_result[key_column] = key
        df_results.append(df_result)

    df_result = pd.concat(df_results)
    return df_result
