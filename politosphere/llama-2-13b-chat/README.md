# Supervised Fine-Tuning LLaMA2-13b-chat-hf on SST2

## Introduction

We have demonstrated how to fine-tune the **LLaMA2-13b-chat-hf** model using QLoRA (Quantized LoRA) here. So we explore the fine-tuned model performances on Politosphere dataset.



## File

- **pre_processing.ipynb**: You can find the politosphere dataset preprocessing steps and the data upload process to HuggingFace in this notebook. We uploaded different datasets depending on the labels.
- **train.py**: You can find the SFT processes in this python file, which supports you train models by epoch or step.
- **evaluate.py**: You can find the model evaluation in this python file. The name of adapters_name is mandatory.

## Usage

```shell
# Train by epoch
python train.py --epoch 50 --max_seq_length 512

# Evaluate model
python evaluate.py --adapters_name Llama-2-13b-chat-hf-epoch10

```



## Data Pre-processing

### Prompt Template

```tex
<s>[INST] <<SYS>>
System prompt
<</SYS>>

User prompt [/INST] Model answer </s>
```

### Example on SST2

```tex
<s>[INST] <<SYS>>
You are a helpful, respectful and honest sentiment analysis assistant. And you are supposed to classify the sentiment of the sentence into one of the following categories: 'positive' or 'negative'.
<</SYS>>

Sentence: hide new secretions from the parental units 
Sentiment: [/INST] positive </s>

```

## Experiment

### Politosphere

**The following experimental content is recommended for reading in Notion Pages. Some experimental images are exclusively presented in [Notion Pages](https://www.notion.so/Project-Politosphere-Overview-3c62083c74d84fac89c9622e9b3c8cb5?pvs=4#3ba252a2769e4494a40285e36fd8950f).**

**Fine-tuning with definitions**

Dataset: daccord_yifei_v2_leo (length of 113)  —> Dataset Description (see README file in the file folder *[data](https://github.com/MeloS7/Politosphere_overview/tree/main/data)*)

—> HuggingFace Dataset [Link](https://huggingface.co/datasets/OneFly7/llama2-politosphere-fine-tuning-system-prompt_with_definition)

(Here we take a train-test split on dataset with ratio 0.8)

Prompt Template: **With Label Definitions.**

```
<s>[INST] <<SYS>> 
You are a sentence sentiment polarity classification assistant about gun control. And here are definitions of labels: Support Gun: Explicitly opposes gun ownership or is in favor of legal policies such as banning guns and confiscating personal guns. Anti Gun: Explicitly in favor of individual gun ownership, or against gun bans and gun confiscation. Neutral: The statement is centered around the debate on gun control, but there is no clear opinion expressed. Not Relevant: Don't have any obvious relationship to guns. Not Sure: The sentence statements are describing gun support for owning / banning guns, but due to a lack of relevant context, or some other reason, we can sense the emotional inclination, but not the specific opinion or polarized aspect. And the sentences are considered as polarized if they are or about antagonizing statements / hostility / belittling / animosity: 'us vs them', inter-group antagonism, radicalization, conflictive confrontation, and so on. the sentences are considered as non-polarized if they are or aboutc onstructive civic conversation, bring together to a common ground, peaceful dialogue, and so on. Please classify the sentiment polarity of the following sentence about gun support into one of the following categories: 'Support Gun Polarized', 'Support Gun non-Polarized', 'Neutral', 'Anti Gun Polarized', 'Anti Gun non-Polarized', 'Not relevant' or 'Not Sure'. 
<</SYS>> 

Sentence: as a federal leo , the very idea of confiscating guns is laughable . i swore an oath to the constitution , not to beto or any other politician . 
User label: [/INST] Support Gun Polarized </s>
```

Train Split: length of 90

Validation Split: length of 23

**Method: Fine-tuning (QLoRA)**

Max_seq_length = 512

Learning Rate = 2e-4

Random seed = 42 (Make it reproducible)

| # Epoch | Accuracy on Train. | F1-score on Train. | Accuracy on Val. | F1-score on Val. |
| ------- | ------------------ | ------------------ | ---------------- | ---------------- |
| 10      | 0.23               | 0.18               | 0.48             | 0.41             |
| 20      | 0.89               | 0.85               | 0.35             | 0.30             |
| 50      | 1.0                | 1.0                | 0.43             | 0.37             |
| 100     | 1.0                | 1.0                | 0.48             | 0.47             |
| 200     | 1.0                | 1.0                | 0.43             | 0.39             |

**P.S.** No invalid output after 20 epochs (including 20).

From the table and graph above, we observe that the model has been trained to be overfitting on this dataset after about 10 epochs. However, the evaluation loss exhibited significant fluctuations between the 20th and 30th epochs, after which it tended to stabilize and even increase.

**I identified an error in my training process—I had not set the random seed for Torch.** Additionally, the language model's generation inherently includes a degree of randomness. Consequently, I hypothesize that this randomness contributes to the observed increase in evaluation loss while still maintaining improvements in accuracy and f1-score.

**For the following experiments, I will set random seed 42 for both training and evaluating process, and set temperature 0.8 to lower the generation randomness.**



**Fine-tuning without definitions**

Dataset: daccord_yifei_v2_leo (length of 113)  —> Dataset Description

—> HuggingFace Dataset [Link](https://www.notion.so/Project-Politosphere-Overview-3c62083c74d84fac89c9622e9b3c8cb5?pvs=21)

(Here we take a train-test split on dataset with ratio 0.8)

Prompt Template: **Without Label Definitions.**

```
"<s>[INST] <<SYS>> 
You are a sentence sentiment polarity classification assistant about gun control. Please classify the sentiment polarity of the following sentence about gun support into one of the following categories: 'Support Gun Polarized', 'Support Gun non-Polarized', 'Neutral', 'Anti Gun Polarized', 'Anti Gun non-Polarized', 'Not relevant' or 'Not Sure'. 
<</SYS>> 

Sentence: as a federal leo , the very idea of confiscating guns is laughable . i swore an oath to the constitution , not to beto or any other politician . 
User label: [/INST] Support Gun Polarized </s>"
```

Train Split: length of 90

Validation Split: length of 23

**Method: Fine-tuning (QLoRA)**

Max_seq_length = 256

Learning Rate = 2e-4

Random seed = 42 (Make it reproducible)

temperature = 0.8

| # Epoch | Accuracy on Train. | F1-score on Train. | Accuracy on Val. | F1-score on Val. |
| ------- | ------------------ | ------------------ | ---------------- | ---------------- |
| 10      | 0.22               | 0.18               | 0.30             | 0.28             |
| 20      | 0.39               | 0.36               | 0.35             | 0.32             |
| 50      | 0.87               | 0.86               | 0.52             | 0.48             |
| 100     | 0.9                | 0.91               | 0.39             | 0.35             |

**P.S.** A Few invalid outputs are generated by all models sometimes.



From the table and graph above, we observe that the model has been trained to be overfitting on this dataset after about 10-15 epochs. However, the evaluation loss exhibited some fluctuations between 15-30 epochs, after which it tended to stabilize and even increase.

In this case, without label definitions in system prompt, the model does **not** perform so well as before with fewer epoch training. But the metrics such as accuracy and f1-score can still attain about 0.50 after 50 epochs, which is similar with the model trained with label definitions.

**Conclusion**

In the aforementioned two experiments, we conducted fine-tuning on the `llama-2-13b-chat-hf` model with and without label definitions, resulting in a multi-class classifier with 7 labels. Through comparison, it was observed that label definitions provided certain assistance to the model's performance during the initial training epochs and in standardizing model outputs.

Our model achieved an f1-score and accuracy close to 0.5, which is considered a commendable outcome for a multi-class classification task (with 7 labels) based on social media data.

In the upcoming experiments, we intend to explore label consolidation. Given the relative ambiguity between some label boundaries, making it challenging for both humans and machines to distinguish, we aim to enhance model performance by reducing the number of label categories.



### Politosphere (polarized/unpolarized/other)

In this section, we firstly merge labels into only 3 labels - **polarized**, **unpolarized** and **other**.

(A script for label merging has been made —> [merge_labels.py](https://github.com/MeloS7/Politosphere_overview/tree/main/script))

We expect to implement a classifier on polarization detection.

We define the new labels as follows (extract some sub-definitions from [here](https://github.com/MeloS7/Politosphere_overview/tree/main/politosphere)):

- Polarized: The sentences are considered as ‘polarized’ if they are or about antagonizing statements / hostility / belittling / animosity: 'us vs them', inter-group antagonism, radicalization, conflictive confrontation, closed to contradiction, trolling, affective, ideological, bad faith, with cognitive bias, with social or demographic bias, irony, sarcasm, hate speech, offensive, toxic, fake news, dismiss language, stereotypes and so on.
- Unpolarized: The sentences are considered as ‘unpolarized’ if they are or about constructive civic conversation, bring together to a common ground, peaceful dialogue, open-minded demeanour focused on learning and forming opinions, genuinely open to contradiction, open exchange of ideas, educational, humble, good faith and so on.
- Other: The sentences are considered as ‘other’ if they not polarized or unpolarized or hard to tell.

**Fine-tuning on pol/unpol/oth**

Dataset: daccord_yifei_v2_leo_pol_unpol_oth (length of 113)  —> Dataset Description

—> HuggingFace Dataset [Link](https://huggingface.co/datasets/OneFly7/llama2-politosphere-fine-tuning-pol-unpol-oth)

(Here we take a train-test split on dataset with ratio 0.8)

Prompt Template: **With Label Definitions.**

```
<s>[INST] <<SYS>>
You are a sentence sentiment polarity classification assistant about gun control. And here are definitions of labels:Polarized: The sentences are considered as 'polarized' if they are or about antagonizing statements / hostility / belittling / animosity: 'us vs them', inter-group antagonism, radicalization, conflictive confrontation, closed to contradiction, trolling, affective, ideological, bad faith, with cognitive bias, with social or demographic bias, irony, sarcasm, hate speech, offensive, toxic, fake news, dismiss language, stereotypes and so on.Unpolarized: The sentences are considered as 'unpolarized' if they are or about constructive civic conversation, bring together to a common ground, peaceful dialogue, open-minded demeanour focused on learning and forming opinions, genuinely open to contradiction, open exchange of ideas, educational, humble, good faith and so on.Other: The sentences are considered as 'other' if they not polarized or unpolarized or hard to tell.Please classify the sentiment polarity of the following sentence about gun support into one of the following categories: 'polarized', 'unpolarized' or 'other'.
<</SYS>>

Sentence: as a federal leo , the very idea of confiscating guns is laughable . i swore an oath to the constitution , not to beto or any other politician .
User label: [/INST] polarized </s>
```

Train Split: length of 90

Validation Split: length of 23

**Method: Fine-tuning (QLoRA)**

Max_seq_length = 512

Learning Rate = 2e-4

Random seed = 42 (Make it reproducible)

temperature = 0.8

| # Epoch | Accuracy on Train. | F1-score on Train. | Accuracy on Val. | F1-score on Val. |
| ------- | ------------------ | ------------------ | ---------------- | ---------------- |
| 10      | 0.73               | 0.71               | 0.48             | 0.49             |
| 20      | 0.98               | 0.98               | 0.70             | 0.69             |
| 50      | 0.98               | 0.98               | 0.70             | 0.71             |
| 100     | 0.98               | 0.98               | 0.74             | 0.75             |

### Politosphere (support/anti/other)

In this section, we firstly merge labels into only 3 labels - **support gun**, **anti gun** and **other**.

(A script for label merging has been made —> [merge_labels.py](https://github.com/MeloS7/Politosphere_overview/tree/main/script))

We expect to implement a classifier for political views on gun control.

We define the new labels as follows (extract some sub-definitions from [here](https://github.com/MeloS7/Politosphere_overview/tree/main/politosphere)):

- **Support Gun**: The sentences are considered as ‘support gun’ if they are explicitly in favor of individual gun ownership, or against gun bans and gun confiscation.
- **Anti Gun**: The sentences are considered as ‘anti gun’ if they are explicitly opposes gun ownership or are in favor of legal policies such as banning guns and confiscating personal guns.
- **Other**: The sentences are considered as ‘other’ if they are not ‘support gun’ or ‘anti gun’ or hard to tell or not relevant to gun control.

**Fine-tuning on supp/anti/other**

Dataset: daccord_yifei_v2_leo_supp_anti_oth (length of 113)  —> Dataset Description

—> HuggingFace Dataset [Link](https://huggingface.co/datasets/OneFly7/llama2-politosphere-fine-tuning-supp-anti-oth)

(Here we take a train-test split on dataset with ratio 0.8)

Prompt Template: **With Label Definitions.**

```
<s>[INST] <<SYS>>
You are a sentence sentiment political tendency classification assistant about gun control. And here are definitions of labels: Support Gun: The sentences are considered as 'support gun' if they are explicitly in favor of individual gun ownership, or against gun bans and gun confiscation. Anti Gun: The sentences are considered as 'anti gun' if they are explicitly opposes gun ownership or are in favor of legal policies such as banning guns and confiscating personal guns. Other: The sentences are considered as 'other' if they are not 'support gun' or 'anti gun' or hard to tell or not relevent to gun control. Please classify the sentiment political tendency of the following sentence about gun support into one of the following categories: 'support gun', 'anti gun' or 'other'.
<</SYS>>

Sentence: as a federal leo , the very idea of confiscating guns is laughable . i swore an oath to the constitution , not to beto or any other politician .
User label: [/INST] support gun </s>
```

Train Split: length of 90

Validation Split: length of 23

**Method: Fine-tuning (QLoRA)**

Max_seq_length = 512

Learning Rate = 2e-4

Random seed = 42 (Make it reproducible)

temperature = 0.8

| # Epoch | Accuracy on Train. | F1-score on Train. | Accuracy on Val. | F1-score on Val. | Comment         |
| ------- | ------------------ | ------------------ | ---------------- | ---------------- | --------------- |
| 10      | 0.60               | 0.50               | 0.70             | 0.63             |                 |
| 20      | 0.47               | 0.54               | 0.39             | 0.43             | Lots of invalid |
| 50      | 0.70               | 0.79               | 0.48             | 0.53             | Lots of invalid |
| 100     | 0.72               | 0.83               | 0.43             | 0.47             | Lots of invalid |

## Author

- Yifei Song (yifei.song@epfl.ch)
