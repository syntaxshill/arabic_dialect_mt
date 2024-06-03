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


## AraT5 translation and evaluation
The following script performs translation using fine-tuned AraT5 model, then sentence-level evaluations and dialect-level evaluations.

We uploaded each of our AraT5 trained checkpoints to huggingface hub, we employed the three stage fine-tuning approach.

To reproduce AraT5 results you can use the eval script for the three stages

```shell
$ python -m src.inference_AraT5 --model_name "ibrahimsharaf/AraT5_stage1" --input_data "data/osact6_task2_test.csv" --batch_size 32
$ python -m src.inference_AraT5 --model_name "ibrahimsharaf/AraT5_stage2" --input_data "data/osact6_task2_test.csv" --batch_size 32
$ python -m src.inference_AraT5 --model_name "ibrahimsharaf/AraT5_stage3" --input_data "data/osact6_task2_test.csv" --batch_size 32
```

AraT5 translations and sentence-level evaluations are stored at `out/{model_name}_{data_file}.csv`.

Dialect-level evaluation is saved at `metrics/{model_name}_{data_file}.csv`.
