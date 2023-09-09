# Supervised Fine-Tuning LLaMA2-13b-chat-hf on SST2

## Introduction

We demonstrate how to fine-tune the **LLaMA2-13b-chat-hf** model using QLoRA (Quantized LoRA) in this repository. We perform sentiment classification on the GLUE SST2 dataset and evaluate our fine-tuning performance.



## File

- **pre_processing.ipynb**: You can find the SST2 dataset preprocessing steps and the data upload process to HuggingFace in this notebook.

- **train.py**: You can find the SFT processes in this python file, which supports you train models by epoch or step.

- **evaluate.py**: You can find the model evaluation in this python file. The name of adapters_name is mandatory.

  

## Usage

```shell
# Train by epoch
python train.py --stop_condition epochs --epoch 50 --max_seq_length 256

# Train by step 
python train.py --stop_condition steps --steps 20 --max_seq_length 256

# Evaluate model
python evaluate.py --adapters_name Llama-2-13b-chat-hf-epoch1

```



## Experiment

| Method              | # Training data | Accuracy | F1-score | Training Loss Variations | Irregular Output  |
| ------------------- | --------------- | -------- | -------- | ------------------------ | ----------------- |
| Fine-tuning (QLoRA) | 100 (1 epoch)   |          |          | 15.9→14.077              | All               |
| Fine-tuning (QLoRA) | 100 (10 epoch)  | 91.40%   | 0.92     | 15.9→0.0706              | Counter({'i': 3}) |
| Fine-tuning (QLoRA) | 100 (20 epoch)  | 93.58%   | 0.94     | 15.9→0.0015              |                   |
| Fine-tuning (QLoRA) | 100 (50 epoch)  | 94.50%   | 0.94     | 15.9→0.001               |                   |
| Fine-tuning (QLoRA) | 500             |          |          | 15.75→4.84               | All               |
| Fine-tuning (QLoRA) | 1000            | 91.97%   | 0.92     | 15.75→0.0958             | Counter({'i': 1}) |
| Fine-tuning (QLoRA) | 5000            | 95.41%   | 0.95     | 15.75→0.0839             |                   |
| Fine-tuning (QLoRA) | 10000           | 96.22%   | 0.96     | 15.75→0.043              |                   |
| Fine-tuning (QLoRA) | Full(60000+)    | 96.56%   | 0.97     | 15.75→0.051              |                   |



## Author

- Yifei Song (yifei.song@epfl.ch)
