# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: fastapi-post接口


import traceback
import logging
import random
import time
import json
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

from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
import torch

from pydantic import BaseModel
from fastapi import FastAPI
import time

from internlm_sft.models.internlm.modeling_internlm import InternLMForCausalLM as LLMForCausalLM
from internlm_sft.models.internlm.tokenization_internlm import InternLMTokenizer as LLMTokenizer
from internlm_sft.ft_internlm.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from internlm_sft.ft_internlm.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from internlm_sft.ft_internlm.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from internlm_sft.ft_internlm.config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from internlm_sft.ft_internlm.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from internlm_sft.ft_internlm.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from internlm_sft.ft_internlm.config import USE_CUDA


app = FastAPI()  # 日志文件名,为启动时的日期, 全局日志格式
logger_level = logging.INFO
logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] "
                           "- %(levelname)s: %(message)s",
                    level=logger_level)
logger = logging.getLogger("ft-chatglm")
console = logging.StreamHandler()
console.setLevel(logger_level)
logger.addHandler(console)


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
def load_model_state(model, model_save_dir="./", model_name="adapter_model.bin", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode = True
        model = get_peft_model(model, peft_config)
        state_dict = torch.load(path_model, map_location=torch.device(device))
        # print(state_dict.keys())
        state_dict = {"base_model.model." + k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print(state_dict.keys())
        print("#" * 128)
        ### 排查不存在model.keys的 state_dict.key
        name_dict = {name: 0 for name, param in model.named_parameters()}
        print(name_dict.keys())
        print("#" * 128)
        for state_dict_key in state_dict.keys():
            if state_dict_key not in name_dict:
                print("{} is not exist!".format(state_dict_key))
        model.load_state_dict(state_dict, strict=False)
        # model.to(device)
        print("******model loaded success******")
        print("self.device: {}".format(device))
    except Exception as e:
        print(str(e))
        raise Exception("******load model error******")
    return model
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
def load_json(path: str, encoding: str="utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json
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

model = LLMForCausalLM.from_pretrained(PATH_MODEL_PRETRAIN)
model = prepare_model_for_half_training(model,
        use_gradient_checkpointing=False,
        output_embedding_layer_name="lm_head",
        layer_norm_names=["post_attention_layernorm",
                          "input_layernorm",
                          "norm",
                          ],
        )
model.is_parallelizable = IS_PARALLELIZABLE
model.model_parallel = MODEL_PARALLEL
model.config.use_cache = USE_CACHE
config = LoraConfig(target_modules=TARGET_MODULES,
                    lora_dropout=LORA_DROPOUT,
                    lora_alpha=LORA_ALPHA,
                    task_type="CAUSAL_LM",
                    bias="none",
                    r=LORA_R,
                    )
model = get_peft_model(model, config)
model = load_model_state(model=model, model_save_dir=MODEL_SAVE_DIR)

if USE_CUDA:
    model = model.float16().cuda()
else:
    model = model.bfloat16()
print_named_parameters(model, True)


def txt_read(path, encode_type="utf-8", errors=None):
    """
        读取txt文件，默认utf8格式, 不能有空行
    Args:
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
        errors[String]: specifies how encoding errors handled, eg. "ignore", strict
    Returns:
        lines[List]: output lines
    """
    lines = []
    try:
        file = open(path, "r", encoding=encode_type, errors=errors)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines
def predict(data_point, generation_config):
    """  推理  """
    prompt_dict = generate_prompt(data_point)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    # input_dict = data_collator([prompt_dict])
    # if USE_CUDA:
    #     input_dict = {k:v.cuda() for k,v in input_dict.items()}
    # print(input_dict)
    generation_config = GenerationConfig(**generation_config)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            # max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(input_ids)
    print(s)
    print(output)
    return output
class Item(BaseModel):
    instruction: str = "完成下面的问答"
    text: str = "1+1="
    penalty_alpha: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.8  # 0.95  # 0.35  # 0.95
    do_sample: bool = True
    num_beams: int = 1
    top_p: float = 0.8  # 0.75
    top_k: int = 50

@app.post("/nlg/text_generate")
def text_generate(request_data: Item):
    instruction = request_data.instruction
    text = request_data.text
    penalty_alpha = request_data.penalty_alpha
    max_new_tokens = request_data.max_new_tokens
    temperature = request_data.temperature
    do_sample = request_data.do_sample
    num_beams = request_data.num_beams
    top_p = request_data.top_p
    top_k = request_data.top_k

    generation_dict = vars(request_data)
    print(generation_dict)
    generation_dict.pop("max_new_tokens")
    generation_dict.pop("instruction")
    generation_dict.pop("text")
    data_point = {"instruction": instruction, "input": text, "output": ""}
    generation_config = {"temperature": temperature,
                         "top_p": top_p,
                         "top_k": top_k,
                         "num_beams": num_beams,
                         "do_sample": do_sample,
                         "penalty_alpha": penalty_alpha,
                         "max_new_tokens": max_new_tokens,
                         "pad_token_id": ID_PAD,
                         "eos_token_id": ID_EOS,

                         }
    try:  # 数据预处理, 模型预测
        response = predict(data_point, generation_config)
    except Exception as e:
        logger.info(traceback.print_exc())
        response = "<eoa>"
    return response


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8036,
                workers=1)


"""
# nohup python post_api.py > tc.post_api.py.log 2>&1 &
# tail -n 1000  -f tc.post_api.py.log
# |myz|

可以在浏览器生成界面直接访问: http://localhost:8032/docs
"""