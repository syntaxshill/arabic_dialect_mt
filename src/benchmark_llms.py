import pandas as pd
import argparse
from pathlib import Path
import tqdm
from .llm_translate import LlmTranslator
from .evaluation import *


def dir_path(path):
    if Path(path).exists and Path(path).is_dir():
        return Path(path)
    else:
        raise NotADirectoryError(path)


# OSACT6 task 2 is Dialect to MSA translation, but we are using LLMs 
# to backtranslate MSA to dialects, so we need to reverse for evaluation
parser = argparse.ArgumentParser(description='Get translations using LLMs and optionally evaluate.')
parser.add_argument('--data_file', help='Input data file.')
parser.add_argument('--output_dir',type=dir_path,  help='Output data dir.')
parser.add_argument('--metric_dir', type=dir_path, help='Metrics output dir.')
# parser.add_argument('--reverse_source', action='store_true', help='Whether to reverse source and target in inputs.')
parser.add_argument('--gpt', action='store_true', help='Whether to use GPT 3.5 model (can only use one model).')
parser.add_argument('--aya', action='store_true', help='Whether to use Aya model (can only use one model).')
parser.add_argument('--gemini', action='store_true', help='Whether to use Gemini model (can only use one model).')
args = parser.parse_args()

model_name = "gpt" if args.gpt else "aya" if args.aya else "gemini"
llm_translator = LlmTranslator(model_name)

# mode = "msa2dia" if args.reverse_source else "dia2msa"
# print(f"Performing {mode} translation with {model_name}")

input_filename = Path(args.data_file).stem
output_filename = f"{model_name}_{input_filename}.csv"
output_file = args.output_dir / output_filename
metric_file = args.metric_dir / output_filename

# load data
df = pd.read_csv(args.data_file)
tqdm.pandas()

print("Starting translation")
df[model_name] = df.progress_apply(lambda row: llm_translator.translate(row["source"], 
                                                               f'{row["dialect"]} Arabic', 
                                                               "Modern Standard Arabic"), axis=1)

print("Evaluating")
df["comet"] = df.progress_apply(lambda row: get_comet_score({
    "src": row["source"],
    "mt": row[model_name],
    "ref": row["target"]
}))

df["bleu"] = df.progress_apply(lambda row: get_bleu_score(row[model_name], [row["target"]]))

df.to_csv(output_file)

print("Performing dialect-level evaluation")
metrics_df = pd.DataFrame(columns=["corpus_bleu", "corpus_comet"])
dialects = df.dialect.unique()
for dialect in dialects:
    dialect_df = df.loc[df['dialect'] == dialect]
    corpus_comet = dialect_df['comet'].mean()

    all_refs = [[x] for x in dialect_df['target']]
    corpus_bleu = get_bleu_score(dialect_df['source'], all_refs, corpus_level=True)
    
    new_row = pd.DataFrame({
        "corpus_bleu": [corpus_bleu],
        "corpus_comet": [corpus_comet]
    })

    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
metrics_df.to_csv(metric_file)
