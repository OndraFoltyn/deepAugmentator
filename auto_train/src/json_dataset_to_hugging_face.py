from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline
import datasets
from datasets import load_dataset, Dataset
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch
import glob
import json
from tqdm import tqdm
import ijson
import decimal

import argparse
import torch
import os
torch.cuda.is_available()

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super().default(obj)

def parse_json_args():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument for the model name
    parser.add_argument('-d', '--json_dataset_paths', nargs='+', default=[], help ='Path for single JSON files (-d "../data/processed/Slavomira/winlog.json" "/path/2")')
    parser.add_argument('-f', '--jsons_folder_paths', nargs='+', default=[], help ='Path for folder contains JSON files (-f "data/datasets/Slavomira/" "/path/2")')
    parser.add_argument('--remove_meta', action='store_true', help ='Use this flag to remove meta field from JSON files')
    parser.add_argument('--remove_mask', action='store_true', help ='Use this flag to remove mask entities from JSON files')
    parser.add_argument('--save_dataset', action='store_true', help ='Use this flag to save the processed dataset to disk')

    parser.add_argument('-o', '--output_path', type=str, required=True)
    return parser.parse_args()


def process_single_file(json_file_path, mask_remove, meta_remove, seed=42):
    print(f"Processing file: {json_file_path}")
    
    # Count total items
    with open(json_file_path, 'rb') as f:
        total_items = sum(1 for _ in ijson.items(f, 'item'))
    
    print(f"File contains {total_items} records")
    
    def record_generator():
        with open(json_file_path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                record = item
                # Remove "word" if present
                if record.get("entities"):
                    for entity in record["entities"]:
                        if "word" in entity:
                            del entity["word"]
                # Remove meta if flag
                if meta_remove and "meta" in record:
                    del record["meta"]
                # Remove mask entities if flag
                if mask_remove:
                    record["entities"] = [e for e in record.get("entities", []) if e.get("entity_group") != "mask"]
                # Keep only 'entities' and 'payload' for consistent schema
                yield {
                    'entities': record.get('entities', []),
                    'payload': record.get('payload', '')
                }
    
    ds = Dataset.from_generator(record_generator)
    ds = ds.shuffle(seed=seed)
    
    if total_items < 2:
        return DatasetDict({"train": ds})
    
    split_ds = ds.train_test_split(test_size=0.2, seed=seed)
    return split_ds


def process_datasets(path, single_file: bool, mask_remove: bool, meta_remove: bool):
    print("Processing single file") if single_file else print("Processing folder with files")
    print(f"Remove meta: {meta_remove}, Remove mask: {mask_remove}")

    train_datasets = []
    test_datasets = []
    path = path if isinstance(path, list) else [path]

    if single_file:
        print("Provided individual JSON files to process.")
        for json_file_path in tqdm(path):
            split_ds = process_single_file(json_file_path, mask_remove, meta_remove)
            train_datasets.append(split_ds["train"])
            if "test" in split_ds:
                test_datasets.append(split_ds["test"])
    else:  
        print("Provided folders with JSON files to process.")
        for jsons_file_path in tqdm(path):
            json_files_from_folder = glob.glob(f"{jsons_file_path}/*.json")
            for json_file_path in tqdm(json_files_from_folder):
                split_ds = process_single_file(json_file_path, mask_remove, meta_remove)
                train_datasets.append(split_ds["train"])
                if "test" in split_ds:
                    test_datasets.append(split_ds["test"])

    print("Dataset processing completed.")

    if not train_datasets:
        raise ValueError("No valid JSON files found in the provided paths.")

    dataset_dict = {"train": concatenate_datasets(train_datasets)}
    if test_datasets:
        dataset_dict["test"] = concatenate_datasets(test_datasets)

    dataset = DatasetDict(dataset_dict)

    return dataset

def save_dataset(dataset, output_path):
    print(f"Saving dataset to {output_path}...")
    dataset.save_to_disk(output_path)
    print("Dataset saved successfully.")

if __name__ == "__main__":
    args = parse_json_args()
    
    if args.json_dataset_paths:
        single_json = True
        path = args.json_dataset_paths
    elif args.jsons_folder_paths:
        single_json = False
        path = args.jsons_folder_paths
    else:
        raise ValueError("You must provide either individual JSON file paths or folder paths containing JSON files.") 
    
    mask_remove = args.remove_mask
    meta_remove = args.remove_meta
    
    dataset = process_datasets(path, single_json, mask_remove, meta_remove)

    if args.save_dataset:
        save_dataset(dataset, args.output_path)