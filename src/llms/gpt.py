import json
import os
import sys
import argparse
import openai
from src import evaluation



openai.api_type = "azure"
openai.api_base = "https://arabic-dialects-llm-translation.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "0395a3266add4e209910eacd74ffe4a2"

def request_gpt(src_text, dialect):
    response = openai.Completion.create(
        engine="gpt-35-turbo-model",
        prompt=f"{dialect} Arabic: {src_text}\nModern Standard Arabic: ",
        temperature=0,
        max_tokens=100,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response


# cut off after newline (may need to do something more involved with other prompts)
def postprocessing(mt_text):
    return mt_text.split('\n')[0] 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get Dialect->MSA translations using GPT and evaluate.')
    parser.add_argument('data_file', help='Input data file.')
    parser.add_argument('output_file', help='Output data file.')
    parser.add_argument('metric_file', help='Metrics output file.')
    # parser.add_argument('--evaluate', action='store_true', help='Whether to calculate BLEU and COMET (requires reference)')

    args = parser.parse_args()

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
            data_by_dialect[dialect] += item

    for dialect, data in data_by_dialect.items():
        print(dialect, len(data))

    # get LLM translations 
    results_by_dialect = {}
    for dialect, data in data_by_dialect.items():
        # data_for_comet = []
        results = []
        for item in data:
            raw_resp = request_gpt(item["source"], dialect)
            raw_text = raw_resp["choices"][0]["text"]
            mt_text = postprocessing(raw_text)

            comet_data = {
                "src": item["source"],
                "mt": mt_text,
                "ref": item["target"]
            }
            bleu_score = evaluation.get_bleu_score(mt_text, [item["target"]])
            comet_score = evaluation.comet_score(comet_data)

            item.update({
                "gpt_translation": mt_text,
                # "bleu_score": bleu_score,
                # "comet_score": comet_score,
            })
            results.append(item)

        results_by_dialect[dialect] = results

    # save results
    with open(args.output_file, "w") as f:
        json.dump(results_by_dialect, f)

    # perform dialect-level evaluation
    metrics_by_dialect = {}
    for dialect, results in results_by_dialect.items():
        all_bleu_scores = [x["bleu_score"] for x in results]
        corpus_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
        
        all_comet_scores = [x["comet_score"] for x in results]
        corpus_comet = sum(all_comet_scores) / len(all_comet_scores)

        metrics_by_dialect[dialect] = {
            "bleu_score": corpus_bleu,
            "comet_score": corpus_comet,
        }
    
    # save metrics
    with open(args.metric_file, "w") as f:
        json.dump(metrics_by_dialect, f)



    


