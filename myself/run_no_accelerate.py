import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from argments import parse_args
from data import CLIRMatrixDataset, CLIRMatrixCollator
from modeling import DaulModel


def main():
    args = parse_args()
    
    # ------------------------ 加载 tokenizer、model、model_config、dataset、dataloader等等 ------------------------
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DaulModel(args).to(device)
    
    
    train_dataset = CLIRMatrixDataset(args=args)
    data_collator = CLIRMatrixCollator(tokenizer, query_max_len=32, document_max_len=128)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    # ------------------------ 加载 tokenizer、model、model_config、dataset、dataloader等等 ------------------------


    # ------------------------ Optimizer 优化器 ------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ------------------------ Optimizer 优化器 ------------------------

    # Train!
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size = {args.per_device_train_batch_size}")
 
    completed_steps = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if completed_steps % 100 == 0:
                print(f'step: {completed_steps}, loss: {loss}')
            completed_steps += 1

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                epoch_save_path = os.path.join(args.output_dir, 'model.pth')
            torch.save(model.state_dict(), epoch_save_path)

    if args.output_dir is not None:
        model_save_path = os.path.join(args.output_dir, 'model.pth')
        torch.save(model.state_dict(), model_save_path)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
