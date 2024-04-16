import os
import re
import pdb
import json
import pandas as pd
import datasets
import promptsource.utils
import numpy as np
from scipy import stats
from main import configure, run


def show_log(output_dir, subset="eval_loss"):
    with open(os.path.join(output_dir, "trainer_state.json"), "r") as f:
        log_history = json.load(f)["log_history"]
    df_log = pd.DataFrame(log_history).dropna(subset=subset).dropna(how='all', axis=1)
    return df_log


def write_eval_results(output_dir, test_dataset, dir_dataset=None, seed=None, each_eval_samples=256):
    path_dataset = os.path.join(dir_dataset, test_dataset) if dir_dataset is not None else test_dataset
    if seed is None:
        argv = f'''
        --do_train False
        --model_name_or_path {output_dir}
        --eval_dir {path_dataset}
        '''
    else:
        argv = f'''
        --do_train False
        --model_name_or_path {output_dir}
        --eval_dir {path_dataset}
        --each_eval_samples {each_eval_samples}
        --seed {seed}
        '''
    trainer, data_args = configure(argv)
    pred_results = run(trainer, data_args)

    fname_json = f"eval_results_{test_dataset}.json" if seed is None else f"eval_results_{test_dataset}_{seed}.json"
    with open(os.path.join(output_dir, fname_json), 'w') as f:
        json.dump(pred_results, f)
    
    return pred_results


# +
def read_eval_results(output_dir_format, test_dataset, dataset_names, prefix="eval_loss_"):
    dict_result = {}
    for dataset_name in dataset_names:
        output_dir = output_dir_format.format(dataset_name)
        with open(os.path.join(output_dir, f"eval_results_{test_dataset}.json"), "r") as f:
            eval_results = json.load(f)
        dict_result[dataset_name] = eval_results

    df_result = pd.DataFrame.from_dict(dict_result, orient="index")
    if prefix == "eval_loss_":
        df_result = df_result[[column for column in df_result.columns if column.startswith(prefix)]]
        df_result = df_result.rename(columns=lambda column: column.replace(prefix, ""))
    else:
        df_result = df_result[[prefix]]
    df_result = df_result.sort_index(axis=1)
    return df_result

def read_epoch(output_dir_format, epoch, dataset_names, prefix="eval_loss_"):
    dict_result = {}
    for dataset_name in dataset_names:
        output_dir = output_dir_format.format(dataset_name)
        df_log = show_log(output_dir)
        df_log = df_log.set_index("epoch")
        dict_result[dataset_name] = pd.concat([df_log.loc[epoch]])

    df_result = pd.DataFrame.from_dict(dict_result, orient="index")
    if prefix == "eval_loss_":
        df_result = df_result[[column for column in df_result.columns if column.startswith(prefix)]]
        df_result = df_result.rename(columns=lambda column: column.replace(prefix, ""))
    else:
        df_result = df_result[[prefix]]
    df_result = df_result.sort_index(axis=1)
    return df_result

def read_epoch_reverse(output_dir_format, init_dir, test_dataset, epoch, dataset_names, prefix="eval_loss_"):
    dict_result = {}
    for dataset_name in dataset_names:
        output_dir = output_dir_format.format(dataset_name)
        df_log = show_log(output_dir)
        df_log = df_log.set_index("epoch")
        dict_result[dataset_name] = pd.concat([df_log.loc[epoch]])

    df_result = pd.DataFrame(dict_result)
    df_result = df_result.loc[[index for index in df_result.index if index.startswith(prefix)]]
    
    with open(os.path.join(init_dir, f"eval_results_{test_dataset}.json"), "r") as f:
        init_results = json.load(f)
    series_init = pd.Series(init_results)
    series_init = series_init.loc[[index for index in series_init.index if index.startswith(prefix)]]
    df_result = df_result.apply(lambda column: column-series_init, axis=0)
    
    df_result = df_result.rename(index=lambda index: index.replace(prefix, ""))
    df_result = df_result.sort_index(axis=1)
    return df_result

def read_log(output_dir_format, dataset_names, key_column, prefix="eval_loss_"):
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
    
    dict_result = list(dict_results.values())[0]
    dict_result.loc[:, "dataset0"] = 0
    dict_results[0] = dict_result
    
    return dict_results

def read_log_reverse(output_dir_format, dataset_names, key_column, init_dir, test_dataset, prefix="eval_loss_"):
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
    
    with open(os.path.join(init_dir, f"eval_results_{test_dataset}.json"), "r") as f:
        init_results = json.load(f)
    series_init = pd.Series(init_results)
    series_init = series_init.loc[[index for index in series_init.index if index.startswith(prefix)]]
    series_init = series_init.rename(index=lambda index: index.replace(prefix, ""))
    
    dict_result = list(dict_results.values())[0]
    df_init = pd.DataFrame({column: series_init for column in dict_result.columns})
    df_init.index.name = dict_result.index.name
    
    dict_results[0] = df_init    
    dict_results = {key: df_result.apply(lambda column: column-series_init, axis=0) for key, df_result in dict_results.items()}
    return dict_results


# +
# def compute_metrics(df_loo, df_log, epoch, method, index):
#     def compute_mrr(df_loo, df_log):
#         indices = np.where(df_loo.rank(ascending=False).values==1)
#         ranks = df_log.rank(ascending=False).values[indices]
#         mrr = np.mean(1/ranks)
#         return mrr

#     metrics = []
#     for column in df_loo.columns:
#         metric = {
#             "pearsonr": stats.pearsonr(df_loo[column], df_log[column])[0], 
#             "spearmanr": stats.spearmanr(df_loo[column], df_log[column])[0],
#             "epoch": epoch,
#             "method": method,
#             "group": column,
#             "index": index,
#         }
#         metrics.append(metric)
        
#     return metrics
# -

def compute_metrics(df_loo, df_log, metric):
    def compute_mrr(df_loo, df_log):
        indices = np.where(df_loo.rank(ascending=False).values==1)
        ranks = df_log.rank(ascending=False).values[indices]
        mrr = np.mean(1/ranks)
        return mrr

    metric["pearsonr"] = np.mean([stats.pearsonr(df_loo[column], df_log[column])[0] for column in df_loo.columns])
    metric["spearmanr"] = np.mean([stats.spearmanr(df_loo[column], df_log[column])[0] for column in df_loo.columns])

    metrics = [metric]
    return metrics


def apply_template(dataset, template, num_proc=None):
    def map_fn(ex):
        ex = promptsource.utils.removeHyphen(ex)
        answer_choices = template.get_answer_choices_list(ex)
        
        try:
            inputs_and_targets = template.apply(ex)
            
            if len(inputs_and_targets) == 2:
                inputs, targets = inputs_and_targets
                if targets == "":
                    ex = {"inputs_pretokenized": inputs, "targets_pretokenized": ""}
                else:
                    ex = {"inputs_pretokenized": inputs, "targets_pretokenized": targets}
            # When template results in an empty example, template.apply returns [""]
            # Also, if the template gets split wrong, len can be > 2
            # We will filter these out later
            else:
                ex = {"inputs_pretokenized": "", "targets_pretokenized": ""}

        except Exception as e:
            print(template.name, e)
            ex = {"inputs_pretokenized": "", "targets_pretokenized": ""}

        if answer_choices:
            ex["choices_pretokenized"] = answer_choices

        return ex

    def filter_fn(ex):
        return len(ex["inputs_pretokenized"]) > 0 and len(ex["targets_pretokenized"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn, num_proc=num_proc).filter(filter_fn, num_proc=num_proc)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs_pretokenized", "targets_pretokenized", "choices_pretokenized"})


def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(dataset_name)
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)
