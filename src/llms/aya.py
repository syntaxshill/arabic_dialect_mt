import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pathlib import Path

root = Path().cwd()


# checkpoint = "CohereForAI/aya-101" 
# aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)
# checkpoint = "CohereForAI/aya-23-8B"
checkpoint = root / "models/aya-23-8b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
aya_model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)


def aya_translate(prompt):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    outputs = aya_model.generate(
        input_ids, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.1,
    )
    # input_ids = input_ids.to(aya_model.device)
    prompt_len = len(input_ids[0])
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def postprocess(text):
    return text.strip()


def aya_translate_seq2seq(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(aya_model.device)
    outputs = aya_model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


def postprocess_seq2seq(text):
    stripped = re.sub(r'<pad>|</s>', '', text)
    return stripped.strip()
