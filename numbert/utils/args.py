from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS)}
    )
    model_type: str = field(metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pre-trained models downloaded from s3"}
    )


@dataclass
class DataProcessingArguments:
    task_name: str = field(
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(processors.keys())}
    )
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertaining to numBERT
    """
    use_tfrecord: bool = field(
        default=False, metadata={"help": "Use tfrecords instead of regular caching to reduce memory load"}
    )
    is_duoBERT: bool = field(default=False, metadata={"help": "Use duoBERT"})
    do_test: bool = field(default=False, metadata={"help": "Whether to run eval on test set."})
    in_batch_negative: bool = field(
        default=False,
        metadata={
            "help": "Whether to sample sequentially so as to maintain both"
            "positive and negative example in batch (even total batch size required)."
        },
    )
    encode_batch: bool = field(default=False, metadata={"help": "Use batch encode tokenization"})
    msmarco_output: bool = field(default=False, metadata={"help": "Return MSMARCO output"})
    trec_output: bool = field(default=False, metadata={"help": "Return TREC output"})
    print_loss_steps: int = field(default=50, metadata={"help": "Print loss every X updates steps."})
    num_cores: int = field(default=8, metadata={"help": "Number of TPU cores to use (1 or 8)."})
    metrics_debug: bool = field(default=False, metadata="Whether to print debug metrics.")
    only_log_master: bool = field(default=False, metadata={"help": "Only log master"})