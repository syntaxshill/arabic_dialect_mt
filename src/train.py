"""### Read MADAR data and convert to HuggingFace Datasets"""

import pandas as pd
from datasets import Dataset

madar = pd.read_csv("data/madar.csv")
merged_dict = madar.to_dict(orient='list')
dataset = Dataset.from_dict(merged_dict)

#dataset_train = dataset
#dataset_dev = dataset
dataset_train = dataset.filter(lambda x: x["split"] == "train")
dataset_dev = dataset.filter(lambda x: x["split"] == "dev")

"""### Prepare dataset for training"""
from transformers import AutoTokenizer

checkpoint = "UBC-NLP/AraT5v2-base-1024"
#checkpoint = "checkpoints/1.1/checkpoint-96"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(example):
    source_lang = example["dialect"]
    target_lang = "MSA"

    # Define the prefix based on the source language
    if source_lang == "Egyptian":
    	prefix = "ترجم من اللهجة المصرية إلى اللغة العربية الفصحى: "
    elif source_lang == "Gulf":
    	prefix = "ترجم من اللهجة الخليجية إلى اللغة العربية الفصحى: "
    elif source_lang == "Levantine":
    	prefix = "ترجم من اللهجة الشامية إلى اللغة العربية الفصحى: "
    elif source_lang == "Iraqi":
    	prefix = "ترجم من اللهجة العراقية إلى اللغة العربية الفصحى: "
    elif source_lang == "Magharebi":
    	prefix = "ترجم من اللهجة المغاربية إلى اللغة العربية الفصحى: "

    inputs = prefix + example["source"]
    targets = example["target"]

    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=64, truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = dataset_train.map(preprocess_function, batched=False)
tokenized_dev = dataset_dev.map(preprocess_function, batched=False)

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

"""### Evaluation"""
import evaluate
metric = evaluate.load("sacrebleu")
import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

"""### Training"""

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")


training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints/madar/",
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    warmup_ratio=0.10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=10,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
