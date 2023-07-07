# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:04
# @author  : Mo
# @function: internlm-7b


import random
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from internlm_sft.ft_internlm.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import (prepare_model_for_int8_training, get_peft_model, LoraConfig)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import unwrap_model
from tensorboardX import SummaryWriter
from datasets import load_dataset
import bitsandbytes as bnb
import torch.nn as nn
import transformers
import torch

from internlm_sft.models.internlm.modeling_internlm import InternLMForCausalLM as LLMForCausalLM
from internlm_sft.models.internlm.tokenization_internlm import InternLMTokenizer as LLMTokenizer
from internlm_sft.ft_internlm.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from internlm_sft.ft_internlm.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from internlm_sft.ft_internlm.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from internlm_sft.ft_internlm.config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from internlm_sft.ft_internlm.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from internlm_sft.ft_internlm.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from internlm_sft.ft_internlm.config_ds import DEEPSPEED_CONF


def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.bin"):
    """  仅保存 有梯度 的 模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        config.save_pretrained(model_save_dir)
        # config.to_dict()
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def print_named_parameters(model, use_print_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model
def generate_prompt(data_point, is_logger=False):
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"

    """
        prompt = ""
        for record in history:
            prompt += f"<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"
        if len(prompt) == 0:
            prompt += "<s>"
        prompt += f"<|User|>:{query}<eoh>\n<|Bot|>:"
        return tokenizer([prompt], return_tensors="pt")
    """
    # user/bot, encode()=(BOS + tokens)
    x_prompt_1 = tokenizer.encode("""<|User|>:""")
    x_prompt_2 = tokenizer.encode("""<eoh>\n<|Bot|>:""")[1:]
    len_prompt = len(x_prompt_1) + len(x_prompt_2)  # """<|User|>:<eoh>\n<|Bot|>:"""
    text_1 = data_point.get('instruction', '') + "\n" + data_point.get('input', '')
    text_2 = f"{data_point.get('output', '')}"
    # end with gMASK, <sop>
    x = tokenizer.encode(text_1.replace(" ", ""))
    y = tokenizer.encode(text_2.replace(" ", ""))[1:]
    if len(x) + len(y) + len_prompt > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
        y = y[:MAX_LENGTH_A]
    x = x_prompt_1 + x + x_prompt_2  # """<|User|>:{text_1}<eoh>\n<|Bot|>:"""
    # <eoa>\n
    y += [ID_EOA, ID_BREAK, ID_EOS]  # 以"""<eoa>\n<\s>"""结尾
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_1)
        print(text_2)
        print(out)
    return out
def data_collator(batch):
    # there's probably a way to do this with the tokenizer settings
    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels"))
                    for i in range(len(batch))]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)
        # labels = [-100] * len(x) + y + [-100] * len_padding
        # input_ids = x + y + [ID_PAD] * (len_padding)
        # attention_mask = [0] * len(x) + [1] * (len_max_batch-len(x))
        if tokenizer.padding_side and tokenizer.padding_side == "left":
            labels = [-100] * len_padding + [-100] * len(x) + y
            input_ids = [ID_PAD] * (len_padding) + x + y
            attention_mask = [1] * len_padding + [0] * len(x) + [1] * len(y)
        else:
            labels = [-100] * len(x) + y + [-100] * len_padding
            input_ids = x + y + [ID_PAD] * (len_padding)
            attention_mask = [0] * len(x) + [1] * (len_max_batch-len(x))
        tensor_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_attention_mask.append(tensor_attention_mask)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = {"attention_mask": batch_attention_mask,
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    return input_dict
def dfs_file(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_files = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            path_files.append(file_path)  # 将文件路径添加进列表
    files = list(set(path_files))
    files.sort()  # the same list
    return files


rank = int(os.environ.get("RANK"))
local_rank = int(os.environ.get("LOCAL_RANK"))
print(local_rank)
tokenizer = LLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Allow batched inference
# tokenizer.padding_side = "right"  # Allow batched inference
ID_EOH = 103027
ID_EOA = 103028
ID_BOS = 1
ID_EOS = 2
ID_PAD = 2
ID_BREAK = 364  # "\n"
ID_COLON = 334  # ":"
ID_BOS_1 = 27232  # 以<|Bot|>:结尾处mask, 即">:": 27232,

model = LLMForCausalLM.from_pretrained(PATH_MODEL_PRETRAIN, device_map="auto")
model = prepare_model_for_half_training(model,
        use_gradient_checkpointing=True,
        output_embedding_layer_name="lm_head",
        layer_norm_names=["post_attention_layernorm",
                          "input_layernorm",
                          "norm",
                          ],
        )
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = True
model.model_parallel = True
model.config.use_cache = USE_CACHE
config = LoraConfig(target_modules=TARGET_MODULES,
                    lora_dropout=LORA_DROPOUT,
                    lora_alpha=LORA_ALPHA,
                    task_type="CAUSAL_LM",
                    bias="none",
                    r=LORA_R,
                    )
model = get_peft_model(model, config)
print_named_parameters(model)
model = model.cuda()
print_named_parameters(model)

tensorboardx_witer = SummaryWriter(logdir=MODEL_SAVE_DIR)


### 只有一个train的情况
data = load_dataset("json", data_files=DATA_PATH)
if VAL_SET_SIZE > 0:
    # train_val = data["train"].train_test_split(test_size=min(VAL_SET_SIZE,
    #                     int(len(data["train"])/10000)), shuffle=True, seed=42)
    VAL_SET_SIZE = max(min(VAL_SET_SIZE, int(len(data["train"])/10000)), 1)
    generate_prompt(data["train"][0], is_logger=True)
    train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(generate_prompt)
    val_data = train_val["test"].shuffle().map(generate_prompt)
else:
    generate_prompt(data["train"][0], is_logger=True)
    train_data = data["train"].shuffle().map(generate_prompt)
    val_data = None


class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=True):
        # inputs = {k: v.cuda() for k, v in inputs.items()}
        inputs = {k: v.to("cuda:{}".format(local_rank)) for k, v in inputs.items()}
        output = model(**inputs)  # if contain labels, will calculate loss
        loss = output.loss
        logs = {}
        if torch.distributed.get_rank() == 0:  # 一般用0，当然，可以选任意的rank保存。
            tr_loss_scalar = self._nested_gather(loss.detach()).mean().item()
            logs["loss"] = round(tr_loss_scalar, 4)
            logs["lr"] = self.lr_scheduler.get_last_lr()[0]
            step = self.state.global_step
            for k, v in logs.items():
                tensorboardx_witer.add_scalar(k, v, step)
            self.log(logs)
        return loss


trainer = CustomTrainer(
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #                     tokenizer, pad_to_multiple_of=8,
        #                     return_tensors="pt", padding=True
        #                 ),
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        model=model,
        args=transformers.TrainingArguments(
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            max_grad_norm=1.0,
            logging_steps=20,
            # warmup_steps=382,  # 618
            warmup_ratio=0.01,
            evaluation_strategy="no",
            lr_scheduler_type="cosine", #'constant',  # "cosine",
            logging_first_step=False,
            # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            # eval_steps=SAVE_STEPS if VAL_SET_SIZE > 0 else None,
            save_strategy="steps",
            save_total_limit=32,
            save_steps=SAVE_STEPS,
            # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            # ddp_find_unused_parameters=None,
            local_rank=local_rank,
            gradient_checkpointing=True,
            group_by_length=True,  # group together samples of roughly the same length in training
            output_dir=MODEL_SAVE_DIR,
            optim="adamw_torch",  # "adamw_hf",
            report_to=[],  # ["tensorboard"],  # [], ["wandb"]
            fp16=True,
            deepspeed=DEEPSPEED_CONF,
        )
    )

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


files = dfs_file(MODEL_SAVE_DIR)
files_name_str = str(files)
flag_checkpoint = True if files and "checkpoint" in files_name_str else False
trainer.train(resume_from_checkpoint=flag_checkpoint)

if torch.distributed.get_rank() == 0:  #一般用0，当然，可以选任意的rank保存。
    save_model_state(model=model, config=config, model_save_dir=MODEL_SAVE_DIR)
    print_named_parameters(model, use_print_data=True)  # 查看LoRA层权重是不是为NAN溢出


# nohup python train.py > tc.train.py.log 2>&1 &
# tail -n 1000  -f tc.train.py.log
# |myz|

# CUDA_VISIBLE_DEVICES=0 deepspeed train.deepspeed.py

