import argparse
import os
from pathlib import Path
from tqdm import tqdm
from .llm_translate import LlmTranslator


parser = argparse.ArgumentParser(description='Get translations from MSA to dialects using LLMs.')
parser.add_argument('--data_file', help='Input data file.')
parser.add_argument('--gpt', action='store_true', help='Whether to use GPT 3.5 model (can only use one model).')
parser.add_argument('--aya', action='store_true', help='Whether to use Aya model (can only use one model).')
parser.add_argument('--gemini', action='store_true', help='Whether to use Gemini model (can only use one model).')
args = parser.parse_args()

output_dir = Path("out/")
os.makedirs(output_dir, exist_ok=True)

model_name = "gpt" if args.gpt else "aya" if args.aya else "gemini"
llm_translator = LlmTranslator(model_name)

with open(args.data_file) as f:
    file_read = f.readlines()
    source_text = [line.strip() for line in file_read]

src_lang = "Modern Standard Arabic"
tgt_lang = "Egyptian Arabic"    # just one dialect for now

output_filename = f"{model_name}_egyptian_backtranslated.txt"
output_filepath = output_dir / output_filename
out_f = open(output_filepath, "w")

print(f"Translating to {tgt_lang}")

for src_text in tqdm(source_text):
    mt_text = llm_translator.translate(src_text, src_lang, tgt_lang)
    out_f.write(mt_text+'\n')

out_f.close()