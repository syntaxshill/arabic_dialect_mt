import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from .llm_translate import LlmTranslator
from .evaluation import *
import warnings 
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Get translations from DA to MSA using LLMs and evaluate.')
parser.add_argument('data_file', help='Input data file.')
parser.add_argument('--gpt', action='store_true', help='Whether to use GPT 3.5 model (can only use one model).')
parser.add_argument('--aya', action='store_true', help='Whether to use Aya model (can only use one model).')
parser.add_argument('--gemini', action='store_true', help='Whether to use Gemini model (can only use one model).')
args = parser.parse_args()

output_dir = Path("out/")
metric_dir = Path("metrics/")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)

model_name = "gpt" if args.gpt else "aya" if args.aya else "gemini"
llm_translator = LlmTranslator(model_name)

input_filename = Path(args.data_file).stem
output_filename = f"{model_name}_{input_filename}.csv"
output_file = output_dir / output_filename
metric_file = metric_dir / output_filename

# load data
df = pd.read_csv(args.data_file)
tqdm.pandas()

print("Starting translation...")
df[model_name] = df.progress_apply(lambda row: llm_translator.translate(row["source"], 
                                                               f'{row["dialect"]} Arabic', 
                                                               "Modern Standard Arabic"), axis=1)

# Drop untranslated samples (gpt had 14)
print(f"{sum(df[model_name].isna())} untranslated samples")
df = df.dropna()

print("Evaluating...")
df = do_sentence_evals(df, mt_col=model_name)
df.to_csv(output_file)
print(f"Saved at {output_file}")


print("Performing dialect-level evaluation...")

# Drop content-filtered rows
filtered_mask = df[model_name] == "Content filtered"
print(f"Dropping {sum(filtered_mask)} filtered samples")
df = df.loc[~filtered_mask]

metrics_df = do_dialect_eval(df, mt_col=model_name, all_dialects=True)
    
metrics_df.to_csv(metric_file, index=False)
print(f"Saved at {metric_file}")
