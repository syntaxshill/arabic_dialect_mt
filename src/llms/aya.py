import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


checkpoint = "CohereForAI/aya-101"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


def aya_translate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = aya_model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


def postprocess(text):
    stripped = re.sub(r'<pad>|</s>', '', text)
    return stripped.strip()
