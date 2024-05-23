import json
import argparse
from pathlib import Path
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
parser.add_argument('--reverse_source', action='store_true', help='Whether to reverse source and target in inputs.')
parser.add_argument('--gpt', action='store_true', help='Whether to use GPT 3.5 model (can only use one model).')
parser.add_argument('--aya', action='store_true', help='Whether to use Aya model (can only use one model).')
parser.add_argument('--gemini', action='store_true', help='Whether to use Gemini model (can only use one model).')
args = parser.parse_args()

model_name = "gpt" if args.gpt else "aya" if args.aya else "gemini"
llm_translator = LlmTranslator(model_name)

mode = "msa2dia" if args.reverse_source else "dia2msa"
print(f"Performing {mode} translation with {model_name}")

input_filename = Path(args.data_file).stem
output_filename = f"{model_name}_{mode}_{input_filename}.json"
output_file = args.output_dir / output_filename
metric_file = args.metric_dir / output_filename

# load data
with open(args.data_file) as f:
    data_raw = json.load(f)

# get all dialects
dialects = set()
for item in data_raw:
    dialects.add(item["dialect"])

# parse data and split by dialect for dialect-level statistics
data_by_dialect = {}
for item in data_raw:
    dialect = item.pop("dialect")
    
    if dialect not in data_by_dialect:
        data_by_dialect[dialect] = [item]
    else:
        data_by_dialect[dialect] += [item]

for dialect, data in data_by_dialect.items():
    print(dialect, len(data))

# get LLM translations 
results_by_dialect = {}
for dialect, data in data_by_dialect.items():
    print(f"Translating {dialect} dialect")
    src_lang = "Modern Standard Arabic" if args.reverse_source else f"{dialect} Arabic"
    tgt_lang = f"{dialect} Arabic" if args.reverse_source else "Modern Standard Arabic"
    
    results = []
    for i, item in enumerate(data):
        src_text = item['target'] if args.reverse_source else item['source']   
        ref_text = item['source'] if args.reverse_source else item['target']  

        raw_output = llm_translator.translate(src_text, src_lang, tgt_lang, return_raw=True)
        mt_text = llm_translator.postprocess(raw_output)

        # getting score for each sample for error analysis
        bleu_score = get_bleu_score(mt_text, [ref_text])
        comet_score = get_comet_score([{
            "src": src_text,
            "mt": mt_text,
            "ref": ref_text
        }])

        result = {
            "src_text": src_text,
            "ref_text": ref_text,
            "llm_text_raw": raw_output,
            "llm_translation": mt_text,
            "bleu_score": bleu_score,
            "comet_score": comet_score,
        }
        results.append(result)

        if i % 5 == 0:
            print(f"{i}/{len(data)} done")
            # if model_name == "gpt":
            #     time.sleep(5)

    results_by_dialect[dialect] = results

# save results
with open(output_file, "w") as f:
    json.dump(results_by_dialect, f, indent=2)

# perform dialect-level evaluation
metrics_by_dialect = {}
for dialect, results in results_by_dialect.items():
    all_mt = [x["llm_translation"] for x in results]
    all_refs = [[x["ref_text"]] for x in results]
    corpus_bleu = get_bleu_score(all_mt, all_refs, corpus_level=True)
    
    all_comet_scores = [x["comet_score"] for x in results]
    corpus_comet = sum(all_comet_scores) / len(all_comet_scores)

    metrics_by_dialect[dialect] = {
        "bleu_score": corpus_bleu,
        "comet_score": corpus_comet,
    }

# save metrics
with open(metric_file, "w") as f:
    json.dump(metrics_by_dialect, f, indent=2)






