# Script

- evaluate_annotation.py : To compare two labelled dataset.
    ```shell
    python evaluate_annotation.py --file_name_1 yifei --file_name_2 leo
    ```
- intersect_data.py : To take the intersection between two dataset.
    ```shell
    python intersect_data.py --dataset1 yifei_v2 --dataset2 leo
    ```
- merge_labels.py : To merge original labels into new labels.
    Here I suppose that the original data labels are :
    - 1: "Support Gun Polarized",
    - 2: "Support Gun Non-Polarized",
    - 3: "Neutral",
    - 4: "Anti Gun Polarized",
    - 5: "Anti Gun Non-Polarized",
    - 6: "Not Relevant",
    - 7: "Not Sure"

    ```shell
    python merge_labels.py --file_name_1 daccord_yifei_v2_leo --merged_labels "{1:[1,2], 2:[4,5], 3:[3,6,7]}" --new_labels "{1:'Support', 2:'Anti', 3:'Others'}"
    ```
    - merged_labels: "1:[1,2]" merge "Suppot Gun Polarized" and "Support Gun Non-Polarized" labels into new label 1.
    - new_labels: "1:'Support'" indicates the new label 1 is "Support".

- dataset_analysis.py: To show the statistical analysis and the distribution of a dataset.

    ```shell
    python dataset_analysis.py --file_name daccord_yifei_v2_leo
    ```

- preprocessing_sst2.ipynb : A jupyter notebook to show you how to convert SST2 Dataset into LLaMA Prompt Template and upload to HuggingFace.

- modify_labels.ipynb: A jupyter notebook to change label name in each dataset. For instance, you want to change label name from "others" to "other" manually.

## Attention!
- Be careful with the dataset address, some of which I have specified the address prefix and suffix in the script.