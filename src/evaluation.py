# from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU
from evaluate import load
import torch


bleu = BLEU()
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
