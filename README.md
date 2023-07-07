# InternLM-SFT
InternLM-7B微调/LORA/推理

## 实验(截至20230707)
```python
1. torch>=1.13, transformers>=4.25.1;
2. tokenizer.encode输出为 [1, 真实文本token]
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    ">:": 27232,
    ">": 330,
    "]": 332,
    "<": 333,
    ":": 334,
    "<eoh>": 103027,
    "<eoa>": 103028,
3. prompt为"""<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n<\s>"""
   本项目实现在internlm_sft/ft_internlm/train.py中的generate_prompt函数,
   inputs-labels截取至“<|Bot|>:”中的“>:”
   个人觉得用bot而不是assiant, 可能是因为编码时候assiant会被截成两个token

```

## 环境配置
```shell
transformers>=4.25.1
torch>=1.13
sentencepiece
```

## 微调样例
```shell
地址: internlm_sft/ft_internlm

配置: internlm_sft/ft_internlm/config.py
训练: python train.py
推理: python predict.py
验证: python evaluation.py
接口: python post_api.py
```



## 参考/感谢
 - [https://github.com/InternLM/InternLM](https://github.com/InternLM/InternLM)
 - [https://github.com/yonzhuo/ChatGLM2-SFT](https://github.com/yonzhuo/ChatGLM2-SFT)
 - [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
 - [https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
 - [math23k](https://aclanthology.org/D17-1088)

## 免责申明
本项目相关资源仅供学术研究之用，使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。
 - 大模型权重的详细协议见[InternLM/InternLM](https://github.com/InternLM/InternLM)
