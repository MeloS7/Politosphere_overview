# Dataset

## Dataset list

- annotated_data_{name} : The dataset of 150 examples annotated by {name}.
    - annotated_data_yifei(_v2) : The dataset annotated by Yifei.
    - annotated_data_leo: The dataset annotated by Leo.
    - annotated_data_llama2 : Few-shot inference by question answering without using prompt template.
    - annotated_data_llama2_v2 : Few-shot inference by question answering with prompt template.
    - annotated_data_llama2_v2_context : Few-shot inference by question answering with prompt template, label definitions.
- daccord_{name1}_{name2} : The dataset intersected cross datasets from {name1} and {name2} by labels.
- to_annotate_150_cleaned : The extracted 150 data to annotate.
- xxxxx_{label1_label2_...} : The label merged verison dataset with label1, label2 ,...



## The dataset we used in expermients

- Golden dataset: daccord_yifei_v2_leo, which is generated from the intersection of 

  - annotated_data_yifei_v2

  - annotated_data_leo

- Label Merged datasets:

  - Polarity (daccord_yifei_v2_leo_pol_unpol_oth)
    - polarized
    - unpolarized
    - other
  - Gun political tendency (daccord_yifei_v2_leo_supp_anti_oth)
    - support gun
    - anti gun
    - other
  - Polarity and political tendency (daccord_yifei_v2_leo_5labels_oth)
    - Support Gun pol
    - Support Gun unpol
    - Against Gun pol
    - Against Gun unpol
    - Other
  - Polarity and political tendency (plus Neutral) (daccord_yifei_v2_leo_6labels_keep_neu)
    - Support Gun pol
    - Support Gun unpol
    - Against Gun pol
    - Against Gun unpol
    - Neutral
    - Other

### Dataset Analysis

In this section, our primary focus is the analysis of manually annotated datasets, with a particular emphasis on examining the **distribution of labels**. We present a detailed overview of datasets that are deemed significant during the labelling process and continue to be of interest in subsequent experiments.

Before label merging:

| Support/Percentage   | Support Gun Polarized | Support Gun non-Polarized | Neutral   | Anti Gun Polarized | Anti Gun non-Polarized | Not Sure  | Not relevant | Total |
| -------------------- | --------------------- | ------------------------- | --------- | ------------------ | ---------------------- | --------- | ------------ | ----- |
| Leo                  | 20 / 0.13             | 17 / 0.11                 | 11/ 0.07  | 25 / 0.17          | 8 / 0.05               | 33 / 0.22 | 36 / 0.24    | 150   |
| yifei_v2             | 21 / 0.14             | 14 / 0.09                 | 15 / 0.10 | 27 / 0.18          | 9 / 0.06               | 26 / 0.17 | 38 / 0.25    | 150   |
| llama2_v2            | 15 / 0.10             | 12 / 0.08                 | 23 / 0.15 | 78 / 0.52          | 3 / 0.02               | 16 / 0.11 | 3 / 0.02     | 150   |
| llama2_v2_context    | 18 / 0.12             | 12 / 0.08                 | 19 / 0.13 | 79 / 0.53          | 0 / 0.00               | 20 / 0.13 | 2 / 0.01     | 150   |
| daccord_yifei_V2_leo | 16 / 0.14             | 11 / 0.10                 | 8 / 0.07  | 19 / 0.17          | 5 / 0.04               | 20 / 0.18 | 34 / 0.30    | 113   |

P.S. The dataset llama2_v2_context means that Llama2 model inferences with label definitions.

After label merging based on `daccord_yifei_v2_leo`:

| Support/Percentage                 | Polarized | UnPolarized | Other     | Total |
| ---------------------------------- | --------- | ----------- | --------- | ----- |
| daccord_yifei_v2_leo_pol_unpol_oth | 35 / 0.31 | 16 / 0.14   | 62 / 0.31 | 113   |

| Support/Percentage                 | Support Gun | Anti Gun  | Other     | Total |
| ---------------------------------- | ----------- | --------- | --------- | ----- |
| daccord_yifei_v2_leo_supp_anti_oth | 27 / 0.24   | 24 / 0.21 | 62 / 0.55 | 113   |

| Support/Percentage               | Support Gun Polarized | Support Gun non-Polarized | Anti Gun Polarized | Anti Gun non-Polarized | Other     | Total |
| -------------------------------- | --------------------- | ------------------------- | ------------------ | ---------------------- | --------- | ----- |
| daccord_yifei_v2_leo_5labels_oth | 16 / 0.14             | 11 / 0.10                 | 19 / 0.17          | 5 / 0.04               | 62 / 0.55 | 113   |

| Support/Percentage               | Support Gun Polarized | Support Gun non-Polarized | Anti Gun Polarized | Anti Gun non-Polarized | Neutral  | Other     | Total |
| -------------------------------- | --------------------- | ------------------------- | ------------------ | ---------------------- | -------- | --------- | ----- |
| daccord_yifei_v2_leo_5labels_oth | 16 / 0.14             | 11 / 0.10                 | 19 / 0.17          | 5 / 0.04               | 8 / 0.07 | 54 / 0.48 | 113   |

### Dataset Comparison

| Cohen’s Kappa          | Leo   | yifei_v2 | llama2_v2 | llama2_v2_with_context |
| ---------------------- | ----- | -------- | --------- | ---------------------- |
| Leo                    | X     | X        | X         | X                      |
| yifei_v2               | 0.703 | X        | X         | X                      |
| llama2_v2              | 0.106 | 0.124    | X         | X                      |
| llama2_v2_with_context | 0.093 | 0.129    | 0.792     | X                      |