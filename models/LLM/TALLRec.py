import os
import random
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from .base.data_utils import *
from .base.utils import *
from .base.abstract_RS import AbstractRS
from .base.abstract_data import AbstractData

import transformers
from transformers import EarlyStoppingCallback

from termcolor import colored, cprint
from functools import partial
from sklearn.metrics import roc_auc_score

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""


def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len, train_on_inputs=True):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len)
    if not train_on_inputs:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def compute_metrics(eval_preds):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    return {'auc': auc}

# No 1939, 3782
# Yes 8241, 3869
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
    labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits.softmax(dim=-1)
    logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
    return logits[:, 1][2::3], gold[2::3]

class TALLRec_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)

    def parse_args(self, args):
        super().parse_args(args)
        self.micro_batch_size = args.micro_batch_size
        self.sample_num = args.sample_num
        self.cutoff_len = args.cutoff_len
        self.train_on_inputs = not args.not_train_on_inputs
        self.group_by_length = not args.not_group_by_length
        print(self.train_on_inputs)
        print(self.group_by_length)

    def preperation(self):
        super().preperation()
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size

    def load_model(self):
        exec('from models.LLM.'+ self.model_name + ' import ' + self.model_name)
        print('Model %s loaded!' % (self.model_name))
        self.model = eval(self.model_name + '(self.args)')

        self.process_data()
        print('Data processed!')
    
    def process_data(self):
        self.data.train_data["train"] = self.data.train_data["train"].shuffle(seed=self.seed).select(range(self.sample_num)) if self.sample_num > -1 else self.data.train_data["train"].shuffle(seed=self.seed)
        self.data.train_data["train"] = self.data.train_data["train"].shuffle(seed=self.seed)
        map_func = partial(generate_and_tokenize_prompt, tokenizer=self.model.tokenizer, cutoff_len=self.cutoff_len, train_on_inputs=self.train_on_inputs)
        self.data.train_data = (self.data.train_data["train"].map(map_func))
        self.data.valid_data = (self.data.valid_data["train"].map(map_func))
        print(self.data.train_data)

    def train(self):
        ddp = False
        trainer = transformers.Trainer(
            model=self.model.llm,
            train_dataset=self.data.train_data,
            eval_dataset=self.data.valid_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.micro_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_steps=20,
                num_train_epochs=self.max_epoch,
                learning_rate=self.lr,
                fp16=True,
                logging_steps=8,
                optim="adamw_torch",
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=self.verbose,
                save_steps=self.verbose,
                output_dir=self.base_path,
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="eval_auc",
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=self.group_by_length,
                report_to=None,
                # report_to="wandb" if use_wandb else None,
                # run_name=wandb_run_name if use_wandb else None,
                # eval_accumulation_steps=10,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.model.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            # 这些都是新加的
            compute_metrics=compute_metrics, # 用于评估时算metrics
            preprocess_logits_for_metrics=preprocess_logits_for_metrics, # 计算了输出的logits和gold的auc
            callbacks = [EarlyStoppingCallback(early_stopping_patience=self.patience)] 
        )
        print("finish trainer and start training")
        trainer.train(resume_from_checkpoint=False)

class TALLRec_Data(AbstractData):
    def __init__(self, args):
        super().__init__(args)
    
    def load_data(self):
        # return super().load_data()
        self.train_data = load_dataset("json", data_files=self.data_directory+"/train.json")
        self.valid_data = load_dataset("json", data_files=self.data_directory+"/valid.json")
        self.test_data = load_dataset("json", data_files=self.data_directory+"/test.json")
        cprint('Data loaded!', 'green', attrs=['bold'])


class TALLRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.parse_args(args)
        self.preperation()
        self.load_llm()
        self.config_finetuning()
    
    def parse_args(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.llm_path = args.llm_path
        self.cutoff_len = args.cutoff_len
    
    def preperation(self):
        return
    
    def load_llm(self):
        self.llm = LlamaForCausalLM.from_pretrained(self.llm_path, 
                                                    load_in_8bit=True,
                                                    torch_dtype=torch.float16,
                                                    device_map="auto")
        
        self.llm = prepare_model_for_int8_training(self.llm)

        self.tokenizer = LlamaTokenizer.from_pretrained(self.llm_path)
        cprint('LLM loaded!', 'green', attrs=['bold'])

        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference

    def config_finetuning(self):
        # return
        self.ft_config =  LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, self.ft_config)
        self.llm.print_trainable_parameters()




