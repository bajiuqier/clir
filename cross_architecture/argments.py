import argparse
from pathlib import Path
from transformers import SchedulerType

HOME_DIR = Path.home() / 'Desktop'

def parse_args():
    parser = argparse.ArgumentParser(description="myself argments")
    parser.add_argument(
        "--log_path",
        type=str,
        default=str(HOME_DIR / 'clir' / 'training_logs'),
        help="训练日志文件路径"
    )
    # ------------------------------------- train/val/test data_file -------------------------------------
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(HOME_DIR / 'clir' / 'data_processing' / 'data_file' / 'train_data.jsonl'),
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )

    # ------------------------------------- tokenizer -------------------------------------
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="加载 tokenizer_name ",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    # ------------------------------------- model -------------------------------------
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="加载 模型的config配置",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=str(HOME_DIR / 'clir' / 'models' / 'models--xlm-roberta-base'),
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        default='cls',
        help="pooling_method cls or mean",
    )
    parser.add_argument(
        "--normlized",
        type=bool,
        default=True,
        help="对模型输出的文本向量 进行 normlized 操作",
    )
    parser.add_argument(
        "--similarity_method",
        type=str,
        default='l2',
        help="相似度计算方式 cos、l2",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    # ------------------------------------- 学习率/优化器相关参数 -------------------------------------
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Number of updates steps to accumulate before performing a backward/update pass. 在执行反向传播/更新传递之前累积的更新步骤数 "
            "实现梯度累积 即在更新模型参数之前累积多个小批次 (mini-batch) 的梯度。"
            "这在以下情况下非常有意义："
                "小显存设备上的大批次训练："
                "- 训练大型模型或使用大批次时，单个 GPU 的显存可能不足以处理一次完整的大批次。"
                "- 通过梯度累积，可以在多个小批次上计算梯度，并在累积到足够大的批次后再更新模型参数，从而模拟大批次训练。"
                "提高稳定性和泛化性能："
                "- 使用较大的批次进行训练，通常可以使梯度估计更加稳定，从而有助于模型的收敛和泛化性能。"
                "- 通过梯度累积，可以在不增加实际显存需求的情况下，获得较大批次训练的部分好处。"
            "假设 args.gradient_accumulation_steps = 4 并且每个小批次的大小为 16。"
            "这样 通过梯度累积 模型参数每经过 4 个小批次 (即有效批次大小为 64) 才更新一次 这样可以在不增加显存需求的情况下 模拟更大批次的训练 "
        )
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    # ------------------------------------- model train -------------------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(HOME_DIR / 'clir' / 'myself' / 'output'),
        help="Where to store the final model."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch. 是否应该在每n步结束时保存各种状态 或者在每个 epoch 末尾保存 ",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder. 训练是否从检查点 文件夹 继续",
    )
    parser.add_argument(
        "--with_tracking",
        type=bool,
        default=False,
        help="Whether to enable experiment trackers for logging. 是否启用实验跟踪器进行日志记录 ",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
            "要将结果和日志报告到的集成。"
            '支持的平台有 "tensorboard"、"wandb"、"comet_ml" 和 "clearml"。使用 "all"（默认）向所有集成报告。'
            "只有在传递 --with_tracking 时才适用。"
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        type=bool,
        default=True,
        help="Whether or not to enable to load a pretrained model whose head dimensions are different. 是否启用加载头部维度不同的预训练模型 ",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        type=bool,
        default=False,
        help=(
            "当 low_cpu_mem_usage=True 时,它会先创建一个空白的模型对象,只有在实际加载预训练权重时才会真正构造模型参数张量。这种延迟加载方式可以显著减少模型加载时的内存峰值使用量。 "
            "需要注意的是,low_cpu_mem_usage 也会带来一些额外的计算开销,因为它需要分别构造每一层的参数张量。但对于内存紧张的情况,这点额外开销是可以接受的。"
            "另一方面,如果您有足够的内存,并且更追求加载速度,那么就可以将 low_cpu_mem_usage 设置为 False,以获得更快的预训练模型加载速度。"
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."

    return args

