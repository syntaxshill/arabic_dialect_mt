import pandas as pd
import re
import argparse
from pathlib import Path


def is_path(path):
    if Path(path).exists:
        return Path(path)
    else:
        raise NotADirectoryError(path)

parser = argparse.ArgumentParser(description='Takes csv and adds sentence-level evals, and performs summary eval.')
parser.add_argument('input_file', type=is_path, help='Input data file.')
args = parser.parse_args()

input_filename = args.input_file.stem
log_dir = args.input_file.parents[0]
output_file = log_dir / f"{input_filename}_safety_filter.csv"

with open(args.input_file) as f:
    lines = [l.strip() for l in f.readlines()]
    lines = [l[len(lines[0]):] for l in lines]              # remove container name
    lines = [re.sub(r"^.*\[.*\]", "", l) for l in lines]    # remove tqdm
    lines = [x.strip() for x in lines if x]

i = 0
all_instances = []
while i < len(lines):
    line = lines[i]
    if "Content filtered:" in line:
        filtered_instance = {}
        toks = line.split(':')
        filtered_instance['src_lang'] = toks[1].strip()
        filtered_instance["src_text"] = toks[2].strip()

        i += 2
    
    elif "category: " in line:
        category = line.split("category: ")[-1].strip()
        probability = lines[i+1].split("probability: ")[-1].strip()

        filtered_instance[category] = probability

        i += 2
    
    elif line == ']':
        all_instances.append(filtered_instance)
        i += 1

    else:
        i += 1

df = pd.DataFrame(data=all_instances)
df.to_csv(output_file)

