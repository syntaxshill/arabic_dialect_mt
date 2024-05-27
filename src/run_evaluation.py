import argparse
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .evaluation import *


parser = argparse.ArgumentParser(description='Takes csv and adds sentence-level evals, and performs summary eval.')
parser.add_argument('input_file', help='Input data file.')
args = parser.parse_args()

input_filename = Path(args.data_file).stem
metric_dir = Path("metrics/")
os.makedirs(metric_dir, exist_ok=True)
metric_file = metric_dir / input_filename

df = pd.read_csv(args.input_file)
tqdm.pandas()

print("Performing sentence-level evaluation...")
df = do_sentence_evals(df)
df.to_csv(args.input_file)
print(f"Saved at {args.input_file}")

print("Performing summary evaluation...")
metrics = do_aggregate_eval(df)

with open(metric_file) as f:
    json.dump(metrics, f)
print(f"Saved at {metric_file}")
