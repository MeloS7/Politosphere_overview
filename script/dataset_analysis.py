import argparse
import pandas as pd
from datasets import load_dataset
from collections import Counter

def load_datasets(file_path_1='annotated_data_yifei_v2.json'):
    data_file_folder = '../data/'
    dataset_self_annot_path = data_file_folder + file_path_1

    dataset_self_annot = load_dataset('json', data_files=dataset_self_annot_path, split='train')

    return dataset_self_annot

def data_analysis(dataset):
    df = pd.DataFrame(dataset)
    user_label = df["User label"].str.lower()
    count_res = user_label.value_counts()
    
    total_num = count_res.sum()
    print("The length of dataset: ", total_num)
    
    print("The distribution of labels:")
    print("Label:\tCount\tPercentage")
    for i, v in enumerate(count_res.items()):
        print(f"{v[0]}: {v[1]}, {v[1]/total_num:.2f}")
        

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Compare two datasets")

    path_suffix = '.json'
    # Add arguments for file paths
    parser.add_argument(
        "--file_name",
        help="Path to the self-annotated dataset file",
        default='annotated_data_yifei_v2',
        action="store"
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    file_path = args.file_name + path_suffix

    # Load datasets
    print("We are showing the analysis of the dataset:", args.file_name)
    dataset_self_annot = load_datasets(file_path)

    # Show dataset info
    data_analysis(dataset_self_annot)

if __name__ == "__main__":
    main()