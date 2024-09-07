import argparse
from pathlib import Path
from transformers import SchedulerType

HOME_DIR = Path.home().parent / 'mnt' / 'workspace'

def add_model_args():
    parser = argparse.ArgumentParser(description="model argments")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=str(HOME_DIR / 'clir' / 'models' / 'models--bert-base-multilingual-uncased'),
        help="文本编码器"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=6,
        help="multihead attention 的头数"
    )

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
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    args = parser.parse_args()
    return args

def add_logging_args():
    parser = argparse.ArgumentParser(description="logging argments")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(HOME_DIR / 'clir' / 'HIKE' / 'mbert' / 'training_logs'),
        help="日志存放文件夹"
    )
    
    # parser.add_argument(
    #     "--num_heads",
    #     type=int,
    #     default=6,
    #     help="multihead attention 的头数"
    # )


    args = parser.parse_args()
    return args

def add_training_args():
    parser = argparse.ArgumentParser(description="logging argments")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(HOME_DIR / 'clir' / 'HIKE' / 'mbert' / 'output'),
        help="Where to store the final model."
    )
    parser.add_argument(
        "--train_dataset_name_or_path",
        type=str,
        default=str(HOME_DIR / 'clir' / 'HIKE' / 'data' / 'train_dataset.jsonl'),
        help="数据集"
    )
    parser.add_argument(
        "--test_dataset_name_or_path",
        type=str,
        default=str(HOME_DIR / 'clir' / 'HIKE' / 'data' / 'test_dataset.jsonl'),
        help="数据集"
    )
    parser.add_argument(
        "--test_qrels_file",
        type=str,
        default=str(HOME_DIR / 'clir' / 'HIKE' / 'data' / 'test_qrels.csv'),
        help="数据集"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子值 保证实验的可复现性"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay to use.")
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

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch. 是否应该在每n步结束时保存各种状态 或者在每个 epoch 末尾保存 ",
    )

    args = parser.parse_args()
    return args