import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import pandas as pd
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("checkpoints/1.1/checkpoint-160")
model = AutoModelForSeq2SeqLM.from_pretrained("checkpoints/1.1/checkpoint-160").to("cuda")


def get_prefix(lang):
    if lang == "Egyptian":
        return "ترجم من اللهجة المصرية إلى اللغة العربية الفصحى: "
    elif lang == "Gulf":
        return "ترجم من اللهجة الخليجية إلى اللغة العربية الفصحى: "
    elif lang == "Levantine":
        return "ترجم من اللهجة الشامية إلى اللغة العربية الفصحى: "
    elif lang == "Iraqi":
        return "ترجم من اللهجة العراقية إلى اللغة العربية الفصحى: "
    elif lang == "Magharebi":
        return "ترجم من اللهجة المغاربية إلى اللغة العربية الفصحى: "
    else:
        return ""


def predict(batch):
    sources = batch["source"]
    langs = batch["dialect"]
    prefixes = [get_prefix(lang) for lang in langs]
    inputs = [prefix + source for prefix, source in zip(prefixes, sources)]

    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_inputs.input_ids.to("cuda")
    attention_mask = tokenized_inputs.attention_mask.to("cuda")    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


osact = pd.read_csv("data/osact6_task2_test.csv")
# Perform batch inference
batch_size = 32
osact["translation"] = ""

for i in tqdm(range(0, len(osact), batch_size)):
    batch = osact[i:i+batch_size]
    translations = predict(batch)
    osact.loc[i:i+batch_size-1, "translation"] = translations

osact.to_csv("data/osact_with_translations.csv", index=False)

