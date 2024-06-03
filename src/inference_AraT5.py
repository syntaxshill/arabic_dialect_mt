import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from src.evaluation import do_sentence_evals, do_dialect_eval


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_prefix(dialect):
    if dialect == "Egyptian":
        return "ترجم من اللهجة المصرية إلى اللغة العربية الفصحى: "
    elif dialect == "Gulf":
        return "ترجم من اللهجة الخليجية إلى اللغة العربية الفصحى: "
    elif dialect == "Levantine":
        return "ترجم من اللهجة الشامية إلى اللغة العربية الفصحى: "
    elif dialect == "Iraqi":
        return "ترجم من اللهجة العراقية إلى اللغة العربية الفصحى: "
    elif dialect == "Magharebi":
        return "ترجم من اللهجة المغاربية إلى اللغة العربية الفصحى: "
    else:
        return ""


def predict(batch, model, tokenizer, device):
    sources = batch["source"]
    dialects = batch["dialect"]
    prefixes = [add_prefix(dialect) for dialect in dialects]
    inputs = [prefix + source for prefix, source in zip(prefixes, sources)]

    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_inputs.input_ids.to(device)
    attention_mask = tokenized_inputs.attention_mask.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask,
                                 max_new_tokens=40, do_sample=True, top_k=30,
                                 top_p=0.95)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


def main(args):
    set_seed(42)  # to reproduce results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    dataset_name = args.input_data.split("/")[-1].split(".")[0]
    model_name = args.model_name.split("/")[-1]

    osact = pd.read_csv(args.input_data)
    tqdm.pandas()

    osact[model_name] = ""

    for i in tqdm(range(0, len(osact), args.batch_size)):
        batch = osact[i:i + args.batch_size]
        translations = predict(batch, model, tokenizer, device)
        osact.loc[i:i + args.batch_size - 1, model_name] = translations

    print("Evaluating...")
    df = do_sentence_evals(osact, mt_col=model_name)

    output_file = f"out/{model_name}_{dataset_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved at {output_file}")

    print("Performing dialect-level evaluation...")

    metrics_df = do_dialect_eval(df, mt_col=model_name, all_dialects=True)
    metrics_file = f"metrics/{model_name}_{dataset_name}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved at {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Arabic dialects to MSA using a pretrained model.")
    parser.add_argument("--model_name", type=str, help="Model checkpoint name on huggingface hub.")
    parser.add_argument("--input_data", type=str, help="Path to input dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation.")

    args = parser.parse_args()
    main(args)
