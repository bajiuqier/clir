import logging
import math
import os
import random

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from argments import parse_args
from data import CLIRMatrixDataset, CLIRMatrixCollator
from modeling import DaulModel

logger = get_logger(__name__)

def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_error()

    # 初始化 Accelerator
    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

    # set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # ------------------------ 加载 tokenizer、model、model_config、dataset、dataloader等等 ------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )

    model = DaulModel(args)
    
    train_dataset = CLIRMatrixDataset(args=args)
    data_collator = CLIRMatrixCollator(tokenizer, query_max_len=32, document_max_len=128)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    # ------------------------ 加载 tokenizer、model、model_config、dataset、dataloader等等 ------------------------


    # ------------------------ Optimizer 优化器 ------------------------
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
    )
    # ------------------------ Optimizer 优化器 ------------------------

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    # 我们需要重新计算我们的总训练步数，因为训练数据加载器的大小可能已经改变了。
    # 这个在多设备上才可以看到差别
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # 重新计算训练的epoch数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 检查 checkpointing_steps 是 'epoch' 还是 数字字符串，如果是数字字符串将其转为 int 类型
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("xxxxxxxxxxx", experiment_config)

    # Train!
    # 实际训练时的 bach_size
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    starting_epoch = 0

    # 从 checkpoint 恢复训练
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(args.resume_from_checkpoint)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        # 调用 accelerator.load_state 加载检查点的状态，包括模型权重和优化器状态。
        accelerator.load_state(checkpoint_path)

        # Extract `epoch_{i}` or `step_{i}`
        # 解析检查点名称，以确定从哪个训练阶段恢复
        training_difference = os.path.splitext(path)[0]

        # 如果名称中包含 "epoch"，则提取出 epoch 数并加 1 作为新的 starting_epoch，并计算已完成的训练步数。
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch

        # 如果名称中包含 "step"，则提取步数，计算起始 epoch 和已完成的步数，以及调整后的恢复步数。
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        
        # 从检查点恢复训练
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()

            # 在每个小批次计算损失后，将损失除以 args.gradient_accumulation_steps。
            # 这是因为我们将在多个小批次上累积梯度，所以每个小批次的损失需要缩小，以保持总体梯度的大小一致。
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        # 模型 验证
        # model.eval()
        # samples_seen = 0
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        #     predictions, references = accelerator.gather((predictions, batch["labels"]))
        #     # If we are in a multiprocess environment, the last batch has duplicates
        #     if accelerator.num_processes > 1:
        #         if step == len(eval_dataloader) - 1:
        #             predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
        #             references = references[: len(eval_dataloader.dataset) - samples_seen]
        #         else:
        #             samples_seen += references.shape[0]
        #     metric.add_batch(
        #         predictions=predictions,
        #         references=references,
        #     )

        # eval_metric = metric.compute()
        # logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    # "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )    
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # huggingface 中的 model 才有 save_pretrained 方法，继承 torch.nn.Module 自定义的 model 没有这个方法
        # unwrapped_model.save_pretrained(
        #     args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        # )
        # 保存模型状态字典
        model_save_path = os.path.join(args.output_dir, 'model.pth')
        if accelerator.is_main_process:
            torch.save(unwrapped_model.state_dict(), model_save_path)
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
