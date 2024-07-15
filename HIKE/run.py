import os
import logging
import math
import torch
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler, BertTokenizer

from utils import set_seed
from argments import add_logging_args, add_model_args, add_training_args
from data import MyDataset, DataCollatorForMe
from modeling import HIKE

logger = logging.getLogger(__name__)

def main():
    logging_args = add_logging_args()
    model_args = add_model_args()
    training_args = add_training_args()
    # 创建 SummaryWriter,指定日志文件保存路径
    writer = SummaryWriter(os.path.join(logging_args.log_dir, 'tensorboard_logs'))

    # 按日期命名日志文件
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logging_args.log_dir, f"training_{current_date}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", # 这里设置时间格式
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)
    
    # ------------------------ 加载 tokenizer、model、model_config、dataset、dataloader等等 ------------------------
    tokenizer = BertTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=not model_args.use_slow_tokenizer, trust_remote_code=model_args.trust_remote_code
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = HIKE(model_args=model_args).to(device)
    
    logger.info("  train_dataset生成中ing......")
    dataset = MyDataset(dataset_file=model_args.dataset_name_or_path)
    # 将数据集划分为训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    data_collator = DataCollatorForMe(tokenizer, max_len=256)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.batch_size, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, collate_fn=data_collator, batch_size=training_args.batch_size, drop_last=True
    )
    # ------------------------ 加载 tokenizer、model、model_config、dataset、dataloader等等 ------------------------


    # ------------------------ Optimizer 优化器 ------------------------
    # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 将encoder和其他模块的参数分别收集到不同的参数组，并设置weight_decay
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5, 'weight_decay': training_args.weight_decay},
        {'params': model.multihead_attn.parameters(), 'lr': 1e-3, 'weight_decay': training_args.weight_decay},
        {'params': model.knowledge_level_fusion.parameters(), 'lr': 1e-3, 'weight_decay': training_args.weight_decay},
        {'params': model.language_level_fusion.parameters(), 'lr': 1e-3, 'weight_decay': training_args.weight_decay},
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': training_args.weight_decay}
    ])

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    total_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        # warmup 的步数
        num_warmup_steps=num_update_steps_per_epoch,
        # num_warmup_steps=math.ceil(num_update_steps_per_epoch / 6),
        num_training_steps=total_train_steps,
    )
    # ------------------------ Optimizer 优化器 ------------------------

    # 检查 checkpointing_steps 是 'epoch' 还是 数字字符串，如果是数字字符串将其转为 int 类型
    checkpointing_steps = training_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)


    # Train!
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("  ***** Running training *****")
    logger.info(f"  当前时间: {formatted_now}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.batch_size * 2}")
    logger.info(f"  Total optimization steps = {total_train_steps}")
    logger.info(f"  训练的设备: {device}, 设备编号: {torch.cuda.current_device()}")


    starting_epoch = 0
    completed_steps = 0

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()

        # 从检查点恢复训练
        # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
        #     # We skip the first `n` batches in the dataloader when resuming from a checkpoint
        #     active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        # else:
        #     active_dataloader = train_dataloader
        active_dataloader = train_dataloader

        for batch_idx, batch in enumerate(active_dataloader):
            qd_batch = {k: v.to(device) for k, v in batch['qd_batch'].items()}
            ed_s_batch = {k: v.to(device) for k, v in batch['ed_s_batch'].items()}
            ed_t_batch = {k: v.to(device) for k, v in batch['ed_t_batch'].items()}

            optimizer.zero_grad()

            outputs = model(
                qd_batch=qd_batch,
                ed_s_batch=ed_s_batch,
                ed_t_batch=ed_t_batch
            )

            loss = outputs.loss
            writer.add_scalar('Loss/train', loss.item(), completed_steps)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            completed_steps += 1

            # 每训练 checkpointing_steps 步 保存一次模型
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir_name = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir_name)
                        os.makedirs(output_dir, exist_ok=True)
                        model_status_save_path = os.path.join(output_dir, 'model.pth')
                        torch.save(model.state_dict(), model_status_save_path)
                        logger.info(f"  ----------------------------------------------------------")
                        logger.info(f"  第 {completed_steps} 步已经训练完成  模型保存在 {output_dir}")
                        logger.info(f"  ----------------------------------------------------------")

            if completed_steps % 10 == 0:
                # print(f'------loss: {loss}, learning_rate: {lr_scheduler.get_last_lr()[0]}, steps: {completed_steps}/{total_train_steps}------')
                logger.info(f"  loss: {loss:.4f},\tsteps: {completed_steps}/{total_train_steps},\tepoch: {epoch},\tlearning_rate: {lr_scheduler.get_last_lr()[0]}")
                
            if completed_steps >= total_train_steps:
                break
        # 每个 epoch 保存一次模型
        # if training_args.checkpointing_steps == "epoch" and (epoch+1) % 5 == 0:
        if training_args.checkpointing_steps == "epoch":
            output_dir_name = f"epoch_{epoch}"
            if training_args.output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir_name)
                # 使用 os.makedirs 创建目录，如果目录不存在的话
                os.makedirs(output_dir, exist_ok=True)
                model_status_save_path = os.path.join(output_dir, 'model.pth')
                torch.save(model.state_dict(), model_status_save_path)
                logger.info(f"  ------------------------------------------------")
                logger.info(f"  第 {epoch} 轮已经训练完成  模型保存在 {output_dir}")
                logger.info(f"  ------------------------------------------------")
    
    # 训练完成后保存模型        
    if training_args.output_dir is not None:
        output_dir = os.path.join(training_args.output_dir, 'training_ended')
        os.makedirs(output_dir, exist_ok=True)

        model_status_save_path = os.path.join(output_dir, 'model.pth')
        torch.save(model.state_dict(), model_status_save_path)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"  ------------------------------------------------")
        logger.info(f"  训练完成!  模型和 tokenizer 保存在 {output_dir}")
        logger.info(f"  ------------------------------------------------")
    
    writer.close()



if __name__ == "__main__":
    main()
