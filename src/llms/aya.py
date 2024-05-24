import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint = "CohereForAI/aya-101" 
checkpoint = "CohereForAI/aya-23-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)


def aya_translate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(aya_model.device)
    outputs = aya_model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


def postprocess(text):
    stripped = re.sub(r'<pad>|</s>', '', text)
    return stripped.strip()
