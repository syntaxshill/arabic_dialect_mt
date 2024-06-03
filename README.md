# LLMs for Arabic Dialect Machine Translation

Please use `requirements.txt` to set up a virtual environment with python 3.10.

This project supports multiple translation and training modes, as outlined below.

## LLM translation and evaluation
The following script performs LLM translation, then sentence-level evaluations, and dialect-level evaluations.

The optional command-line arguments `[--gpt] [--aya] [--gemini]` specify which model to use.

LLM translations and sentence-level evaluations are stored at `out/{model_name}_{data_file}.csv`.

Dialect-level evaluation is saved at `metrics/{model_name}_{data_file}.csv`.

```shell
$ python -m src.benchmark_llms data_file [--gpt] [--aya] [--gemini]
```


## AraT5 finetuning and evaluation


## LLM backtranslation, AraT5 finetuning, and evaluation

