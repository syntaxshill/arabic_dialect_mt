import argparse
from pathlib import Path
from tqdm import tqdm
from .llm_translate import LlmTranslator


def dir_path(path):
    if Path(path).exists and Path(path).is_dir():
        return Path(path)
    else:
        raise NotADirectoryError(path)

parser = argparse.ArgumentParser(description='Get translations from MSA to dialects using LLMs.')
parser.add_argument('--data_file', help='Input data file.')
parser.add_argument('--output_dir',type=dir_path,  help='Output data dir.')
parser.add_argument('--gpt', action='store_true', help='Whether to use GPT 3.5 model (can only use one model).')
parser.add_argument('--aya', action='store_true', help='Whether to use Aya model (can only use one model).')
parser.add_argument('--gemini', action='store_true', help='Whether to use Gemini model (can only use one model).')
args = parser.parse_args()

model_name = "gpt" if args.gpt else "aya" if args.aya else "gemini"
llm_translator = LlmTranslator(model_name)

with open(args.data_file) as f:
    file_read = f.readlines()
    source_text = [line.strip() for line in file_read]

src_lang = "Modern Standard Arabic"
for dialect in []:

    output_filename = f"{model_name}_{dialect}_backtranslated.txt"
    output_filepath = args.output_dir / output_filename
    out_f = open(output_filepath, "w")

    tgt_lang = f"{dialect} Arabic"
    print(f"Translating to {tgt_lang}")

    for src_text in tqdm(source_text):
        mt_text = llm_translator.translate(src_text, src_lang, tgt_lang)
        out_f.write(mt_text+'\n')

    out_f.close()