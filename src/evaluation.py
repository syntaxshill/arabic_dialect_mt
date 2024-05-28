# from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU
from evaluate import load
import torch
import pandas as pd


# takes a df and enriches with sentence-level evaluations
def do_sentence_evals(df, mt_col="translation"):
    df["comet"] = get_comet_score(df["source"], df[mt_col], df["target"])
    df["bleu"] = df.progress_apply(lambda row: get_bleu_score(row[mt_col], [row["target"]]), axis=1)

    return df


# takes df and produces corpus-level metrics
def do_aggregate_eval(df, mt_col="translation"):
    if 'comet' in df.columns:
        corpus_comet = df['comet'].mean()
    else:
        corpus_comet = get_comet_score(df["source"],
                                       df[mt_col],
                                       df["target"],
                                       corpus_level=True)

    all_refs = [[x] for x in df['target']]
    corpus_bleu = get_bleu_score(df[mt_col].tolist(), all_refs, corpus_level=True)

    metrics = {
        "corpus_bleu": corpus_bleu,
        "corpus_comet": corpus_comet
    }

    return metrics


# Gets corpus level statistics for each dialect.
# if all_dialects, then also evaluates the whole df.
def do_dialect_eval(df, mt_col="translation", all_dialects=False):
    metrics_df = pd.DataFrame(columns=["dialect", "corpus_bleu", "corpus_comet"])
    dialects = df.dialect.unique()
    for dialect in dialects:
        dialect_df = df.loc[df['dialect'] == dialect]
        
        dialect_metrics = do_aggregate_eval(dialect_df, mt_col)
        dialect_metrics["dialect"] = dialect

        new_row = pd.DataFrame.from_records([dialect_metrics])
        metrics_df = pd.concat([metrics_df, new_row])

    if all_dialects:
        metrics = do_aggregate_eval(df, mt_col)
        metrics["dialect"] = "all"
        new_row = pd.DataFrame.from_records([metrics])
        metrics_df = pd.concat([metrics_df, new_row])

    return metrics_df


bleu = BLEU(effective_order=True)
'''
Returns sentence-level(default) or corpus level scores.
If sentence-level:
    sys is the string being evaluated. 
    refs is a list of reference translations.
If corpus-level:
    sys must be Sequence[str]. 
    refs is a list of lists, one for each sample in sys.
'''
def get_bleu_score(sys, refs, corpus_level=False):
    if corpus_level:
        return bleu.corpus_score(sys, refs).score
    else:
        bleu.effective_order = True
        return bleu.sentence_score(sys, refs).score


NO_OF_GPUs = torch.cuda.device_count()
comet_metric = load('comet')
def get_comet_score(source, hypothesis, reference, corpus_level=False):
    comet_score = comet_metric.compute(
        predictions=hypothesis, 
        references=reference, 
        sources=source, 
        gpus=NO_OF_GPUs, 
        progress_bar=True
    )
    return comet_score["mean_score"] if corpus_level else comet_score["scores"]


# comet_checkpoint = download_model("Unbabel/wmt22-comet-da")
# comet_model = load_from_checkpoint(comet_checkpoint)
# ''' 
# Data must be in the following format:
# data = [
#     {
#         "src": "10 到 15 分钟可以送到吗",
#         "mt": "Can I receive my food in 10 to 15 minutes?",
#         "ref": "Can it be delivered between 10 to 15 minutes?"
#     }
# ]
# '''
# def get_comet_score(data, corpus_level=False):
#     model_output = comet_model.predict(data)
#     return model_output.system_score if corpus_level else model_output.scores
