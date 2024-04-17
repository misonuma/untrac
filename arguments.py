import os
import pdb
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple
from transformers import Seq2SeqTrainingArguments
from transformers.training_args import OptimizerNames, trainer_log_levels
from transformers.utils import ExplicitEnum
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType
from collections import Counter
import re


def args_to_output_dir(argv, ignore_arg=None):
    abbr_dict = {
        "train_adapter": "adapter",
        "per_device_train_batch_size": "train_batch",
        "gradient_accumulation_steps": "accumul",
        "train_dir": "train",
        "retain_dir": "retain",
        "learning_rate": "lr",
        "warmup_steps": "warmup",
        }
    del_list = [
        "do_train",
        "do_eval",
        "generation_max_length",
        "per_device_eval_batch_size",
        "logging_steps",
        "logging_strategy",
        "save_steps",
        "save_strategy",
        "eval_steps",
        "evaluation_strategy",
        "max_steps",
        "num_train_epochs",
        "log_level",
        "logging_strategy",
        "metric",
        "predict_with_generate",
        "save_arnoldi",
        "overwrite_output_dir",
    ]
    
    args = argv.strip().split("--")[1:]
    args = [arg.split("/")[-1] if "/" in arg else arg for arg in args] # model_name_or_path removed
    args = [arg for arg in args if not re.match("|".join(del_list), arg)]
    output_dir = "-".join([arg.strip().replace(" ", "=").replace("model=", "") for arg in args])
    for arg, abbr in abbr_dict.items():
        output_dir = output_dir.replace(arg, abbr)
    return output_dir


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    checkpoint: str = field(
        default=None,
    )
    causal: bool = field(
        default=False, 
    )
    model: Optional[str] = field(
        default=None, 
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
        
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total sequence length after tokenization (used for decoder only model). Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    each_eval_samples: Optional[int] = field(
        default=None,
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
        
    train_dir: List[str] = field(
        default=None,
    )
    eval_dir: List[str] = field(
        default=None,
    )
        
    dataset_names: Optional[List[str]] = field(
        default=None,
    )
    loo_dataset_names: Optional[List[str]] = field(
        default=None,
    )
    eval_dataset_names: Optional[List[str]] = field(
        default=None,
    )
        
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
            
    def __post_init__(self):
        pass


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    # super arguments
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    metric: str = field(
        default=None,
    )
    generation_max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=128,
    )
    eos_token_id: Optional[int] = field(
        default=None,
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
        
    scratch: bool = field(
        default=False, 
    )
        
    unlearn: bool = field(default=False)
    
    lr_scheduler_type: SchedulerType = field(
        default="constant_with_warmup",
        metadata={"help": "The scheduler type to use."},
    )
    logging_steps: int = field(default=10000, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    eval_steps: int = field(default=10000, metadata={"help": "Run an evaluation every X steps."})
    evaluation_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    log_level: Optional[str] = field(
        default="warning",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
        
    # original arguments
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(
        default=False, 
        metadata={"help": "Whether to run the model as a demo in the terminal."}
    )
    report_to: Optional[List[str]] = field(
        default="none", metadata={"help": "The list of integrations to report the results and logs to."}
    )
        
    optim: OptimizerNames = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    learning_rate: float = field(
        default=5e-5, 
        metadata={"help": "The initial learning rate for AdamW."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm."}
    )           
    parallelize: bool = field(
        default=False,
    )
    tmp: bool = field(
        default=False, 
    )
