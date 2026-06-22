# %%
print("Parsing arguments")
from src.cli import parse_args
args = parse_args()

# %%
upper_border = "┏" + "━" * 150 + "┓"
inner  = "┃" + " " * 150 + "┃"
lower_border = "┗" + "━" * 150 + "┛"


print("Importing libraries")
# %%
import shutil
import mlflow.data.huggingface_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, AutoConfig, DataCollatorForTokenClassification
from transformers import pipeline
import datasets
from datasets import Dataset, DatasetDict
import numpy as np
import torch
import evaluate #!pip install seqeval
from tqdm import tqdm
# HF_EVALUATE_OFFLINE=1 #for offline evaluation
metric = evaluate.load("seqeval") #!pip install evaluate
import pickle                         
import json
from dotenv import load_dotenv
from collections import Counter
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
load_dotenv()
from functools import partial
import math
import time


from sectech_models.trainer import SectechTrainer
from sectech_models.lstmner import LSTMNERConfig, LSTMNERModel, LSTMNERTokenizer
from datasets import DatasetDict, Dataset
from collections import Counter

from datasets import Dataset

import os
import wandb
import sys
import traceback

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from src.training_functions import *
from src.json_dataset_to_hugging_face import *

print("Libraries imported successfully.\n")

print("-"  * 150)

print("\nParsed string command-line arguments:")
print("model_name:", args.model_name)

if args.dataset_path:
    print("dataset_path:", args.dataset_path)
elif args.json_dataset:
    print("json_dataset:", args.json_dataset)
elif args.dataset_folder:
    print("dataset_folder:", args.dataset_folder)

print("base_dir:", args.base_dir)
print("model_tokenizer_cache_path:", args.model_tokenizer_cache_path)
print("default experiment_id:", args.experiment_id)
print("num_epochs:", args.num_epochs)
print("filter_classes:", args.filter_classes)
print("path_filter_classes:", args.path_filter_classes)
print("remove_class_from_json:", args.remove_class_from_json)
print("vocab_size:", args.vocab_size)
print("download_for_retrain:", args.download_for_retrain)
print("train_batch_size:", args.train_batch_size)
print("eval_batch_size:", args.eval_batch_size)
print("run_name:", args.run_name)

print("\nParsed boolean command-line arguments:")
print("use_mlm_model:", args.use_mlm_model)
print("augmentate:", args.augmentate)
print("augmentate_valid:", args.augmentate_valid)
print("disable_mlflow:", args.disable_mlflow)
print("concat_train_test:", args.concat_train_test)
print("use_fp16:", args.use_fp16)
print("use_class_weight:", args.use_class_weight)
print("no_calc_per_class_f1:", args.no_calc_per_class_f1)
print("do_not_evaluate:", args.do_not_evaluate)
print("val_2_times:", args.val_2_times)
print("generate_synthetic_train:", args.generate_synthetic_train)
print("generate_synthetic_valid:", args.generate_synthetic_valid)
print("train_tokenizer:", args.train_tokenizer)
print("make_custom_tokenizer:", args.make_custom_tokenizer)
print("preprocess:", args.preprocess)
print("make drain:", args.drain)
print("add_special_tokens_and_normalize:", args.add_special_tokens_and_normalize)

print("\nParsed other command-line arguments:")
print("random_seed:", args.random_seed)
print("tracking_uri:", args.tracking_uri)
print("synthetic_train_samples:", args.synthetic_train_samples)
print("synthetic_valid_samples:", args.synthetic_valid_samples)
print("shuffle_seed:", args.shuffle_seed)
print("max_length:", args.max_length)
print("num_layers_unfreeze:", args.num_layers_unfreeze)
print("weight_decay:", args.weight_decay)
print("logging_steps:", args.logging_steps)
print("eval_accumulation_steps:", args.eval_accumulation_steps)
print("metric_for_best_model:", args.metric_for_best_model)
print("greater_is_better:", args.greater_is_better)
print("dataloader_num_workers:", args.dataloader_num_workers)

USE_MLFLOW = not args.disable_mlflow
CALC_PER_CLASS_F1 = not args.no_calc_per_class_f1
DO_NOT_EVALUATE = args.do_not_evaluate
PREPROCESS_DATASET = args.preprocess
DRAIN_DATASET = args.drain
NORMALIZE_ENTITIES = args.add_special_tokens_and_normalize
DOWNLOAD_FOR_RETRAIN = args.download_for_retrain
AUGMENTATE = args.augmentate
ADD_SPECIAL_TOKENS = args.add_special_tokens_and_normalize
TRAIN_BATCH_SIZE = args.train_batch_size
EVAL_BATCH_SIZE = args.eval_batch_size
dataset_path = args.dataset_path
num_epoch = args.num_epochs
num_proc = args.dataloader_num_workers
model_name = args.model_name
use_mlm_model = args.use_mlm_model
model_tokenizer_cache_path = args.model_tokenizer_cache_path
base_dir = args.base_dir

def _handle_fatal_exception(e, context_msg: str = None):
    """Print the error with traceback, mark MLflow run as FAILED if active, then re-raise."""
    if context_msg:
        print(f"FATAL ERROR ({context_msg}): {e}")
    else:
        print(f"FATAL ERROR: {e}")
    traceback.print_exc()
    try:
        if USE_MLFLOW:
            try:
                mlflow.end_run(status="FAILED")
                print("MLflow run ended and marked as FAILED")
            except Exception as end_err:
                print(f"Unable to end MLflow run cleanly: {end_err}")
    except NameError:
        # mlflow not defined / not imported
        pass
    # Re-raise to ensure process exits with error
    raise

if dataset_path:
    dataset_name = dataset_path.split("/")[-1]
elif args.json_dataset:
    dataset_name = args.json_dataset.split("/")[-1].replace(".json", "")
elif args.dataset_folder:
    dataset_name = args.dataset_folder.split("/")[-1]
else:
    dataset_name = "synthetic"
print("dataset_name:", dataset_name)

print("Is Cuda available:", torch.cuda.is_available())

if "t5" in model_name.lower() or "canine" in model_name:
    args.use_fp16 = False
    use_bf16=True   # Enable bfloat16 precision for models pretrained on google TPU
else:
    use_bf16=False

# Strip the model name if it contains a slash
stripped_model_name = model_name.split("/")[-1]

# Define the run name
if args.run_name is not None:
    run_name = args.run_name
else:
    run_name = f"sectech_{stripped_model_name}_{dataset_name}_aug-synthetic" if args.augmentate else f"sectech_{stripped_model_name}_{dataset_name}_no_aug"

    if PREPROCESS_DATASET:
        run_name = f"sectech_{stripped_model_name}_preprocessed_{dataset_name}_aug-synthetic" if args.augmentate else f"sectech_{stripped_model_name}_preprocessed_{dataset_name}_no_aug"
    else: 
        run_name = run_name

    if DRAIN_DATASET:
        run_name = f"sectech_{stripped_model_name}_drained_{dataset_name}_aug-synthetic" if args.augmentate else f"sectech_{stripped_model_name}_drained_{dataset_name}_no_aug"
    else: 
        run_name = run_name


    # Append "_fp16" to run_name if fp16 precision is enabled
    if args.use_fp16:
        run_name = f"{run_name}_fp16"
        
    if use_bf16:
        run_name = f"{run_name}_bf16"

    if args.use_class_weight:
        run_name = f"{run_name}_class_weight"

    if args.augmentate_valid:
        run_name = f"{run_name}_val_aug"

    if args.train_tokenizer:
        run_name = f"{run_name}_t_tokenizer"


print("\nRun name:", run_name)

train_batch_size = TRAIN_BATCH_SIZE
eval_batch_size = EVAL_BATCH_SIZE

run_id = None

if USE_MLFLOW:
    print("Importing MLflow")
    import mlflow
    import mlflow.data

    print("MLflow version:", mlflow.__version__)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", args.tracking_uri)
    # Set tracking server URI
    mlflow.set_tracking_uri(uri=tracking_uri)
    print("Tracking URI:", mlflow.get_tracking_uri())
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    # Set the experiment ID
    if args.experiment_name is not None:
        experiment_id = get_experiment_id(args, client)
    else:
        experiment_id = args.experiment_id
        print(f"Using MLflow experiment ID: {experiment_id}")

    print()
    print(upper_border)
    print(inner)
    print("┃{:^150}┃".format("STARTING MLFLOW RUN"))
    print(inner)
    print(lower_border)

    mlflow.start_run(experiment_id=experiment_id, run_name=run_name, log_system_metrics=True)
    
    run_id = mlflow.active_run().info.run_id
    client.set_tag(run_id, args.tag_key, args.tag_value)

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("dataset", dataset_path)
    mlflow.log_param("epochs", num_epoch)
    mlflow.log_param("train_batch_size", train_batch_size)
    mlflow.log_param("eval_batch_size", eval_batch_size)
    mlflow.log_param("weight_decay", args.weight_decay)
    mlflow.log_param("warmup_steps", args.warmup_steps)

    safe_params = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in vars(args).items()}
    mlflow.log_params(safe_params)
else:
    print("MLflow tracking is disabled.\n")

print("\nRun ID:", run_id if USE_MLFLOW else "Run ID: N/A")

# 1) Directory / environment setup
try:
    # Create a new directory for the model
    if run_id is not None:
        base_dir = os.path.join(base_dir, run_id)
    else:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        timestamp = datetime.now(ZoneInfo("Europe/Prague")).strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")

    os.makedirs(base_dir, exist_ok=True)

    validation_dir = os.path.join(base_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)

    # Create 'results' and 'logs' directories within the new directory
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logging_dir, exist_ok=True)

    eval_output_dir = os.path.join(base_dir, 'eval_outputs')
    os.makedirs(eval_output_dir, exist_ok=True)

    console_output_dir = os.path.join(base_dir, 'console_output')
    os.makedirs(console_output_dir, exist_ok=True)

    mlm_model_dir = os.path.join(base_dir, 'mlm_model')

    if use_mlm_model and not os.path.exists(mlm_model_dir):
        print("MLM model directory does not exist")
        sys.exit(1)

    files_dir = os.path.join(base_dir, 'files')
    os.makedirs(files_dir, exist_ok=True)
except Exception as e:
    _handle_fatal_exception(e, context_msg="environment setup")

# 2) Augmentator and dataset loading
try:
    # Initialize augmentator separately
    print("Initializing augmentator")
    augmentator = initialize_augmentator(seed = args.random_seed)
    print()

    print("-"  * 150)

    from src.data import load_dataset
    dataset = load_dataset(args)

    print("\nDataset type:", type(dataset))
    print(dataset)

    print("\nTrain:")
    print("Train set size:", len(dataset["train"]))
    print("\nTrain dataset column names:", dataset["train"].column_names)

    print("\nTest:")
    print("Test set size:", len(dataset["test"]))
    print("\nTest dataset column names:", dataset["test"].column_names)

    if len(dataset["train"]) < 10:
        sample_range = len(dataset["train"])
    else:
        sample_range = 10

    train_set_for_logging = dataset["train"]

    # Limit test dataset na max 100k pro evaluaci, zbývající přesun do train
    max_eval_size = 100000
    is_dataset_too_large = False

    if len(dataset["test"]) > max_eval_size:
        is_dataset_too_large = True

    # if len(dataset["test"]) > max_eval_size:
    #     print()
    #     print("-"  * 150)
    #     print(f"Test dataset too large ({len(dataset['test'])}), re-splitting to 90/10 train/test ratio.")
    #     combined = datasets.concatenate_datasets([dataset["train"], dataset["test"]])
    #     new_split = combined.train_test_split(test_size=0.1, seed=args.shuffle_seed)
    #     dataset = new_split
    #     train_set_for_logging = dataset["train"]
    #     print("\nNew train set size:", len(dataset["train"]))
    #     print("New test set size:", len(dataset["test"]))
    # else:
    #     print(f"Test dataset size ({len(dataset['test'])}) is within the limit, no re-splitting needed.")

    eval_accumulation_steps = args.eval_accumulation_steps
    eval_batch_size_threshold = 4
    eval_accumulation_steps_threshold = 10

    # Dynamicky uprav eval_batch_size pro velké eval datasety (např. > 100k vzorků)
    if len(dataset["test"]) > max_eval_size and args.eval_batch_size > eval_batch_size_threshold:
        eval_batch_size = min(eval_batch_size, eval_batch_size_threshold)  # Snížit na max 4 pro velké datasety
        print(f"\nAdjusted eval_batch_size to {eval_batch_size} due to large eval dataset (size: {len(dataset['test'])})")

    # Dynamicky uprav eval_accumulation_steps pro lepší správu paměti
    if len(dataset["test"]) > (max_eval_size // 2) and eval_accumulation_steps < eval_accumulation_steps_threshold:
        eval_accumulation_steps = max(eval_accumulation_steps, eval_accumulation_steps_threshold)  # Zvětšit na min 10 pro velké datasety
        print(f"eval_accumulation_steps is set to {eval_accumulation_steps} due to large eval dataset\n")

except Exception as e:
    _handle_fatal_exception(e, context_msg="augmentator/dataset loading")

# 3) Data preprocessing and filtering
try:
    if PREPROCESS_DATASET:
        print("-"  * 150)
        dataset, log_processor = preprocess_dataset(dataset)

    if DRAIN_DATASET:
        print("-"  * 150)
        dataset = make_drain_dataset(dataset)

    if NORMALIZE_ENTITIES:
        print("-"  * 150)
        print("\nNormalizing entities in the dataset samples:")
        disable_placeholders = ["DOMAIN", "FILEPATH"]
        normalize_dataset_samples = dataset["train"].select(range(sample_range))
        normalize_dataset_samples = normalize_tokens_in_dataset(normalize_dataset_samples, disable_placeholders)

        for i in range(len(normalize_dataset_samples)):
            print(f"\nSample {i}:")
            if "payload" in normalize_dataset_samples:
                print("Payload:", normalize_dataset_samples["payload"][i])

    # Filtering classes if requested
    if args.path_filter_classes:
        if args.remove_class_from_json:
            print(
                f"\nFiltering out classes {args.remove_class_from_json} "
                f"from categories.json at {args.path_filter_classes}"
            )
            classes_to_keep = remove_classes_from_json(
                args.path_filter_classes,
                args.remove_class_from_json
            )
        else:
            classes_to_keep = load_classes_from_json(args.path_filter_classes)
        print("Classes to keep loaded from JSON:", classes_to_keep)
    elif args.filter_classes:
        classes_to_keep = args.filter_classes
    else:
        classes_to_keep = None

    if classes_to_keep is not None:
        # (keeps the existing filtering logic)
        cleaned_classes = []
        for class_name in classes_to_keep:
            if class_name == 'O':
                continue
            if class_name.startswith('B-') or class_name.startswith('I-'):
                cleaned_classes.append(class_name[2:])
            else:
                cleaned_classes.append(class_name)

        # Remove duplicates by converting to set and back to list
        cleaned_classes = list(set(cleaned_classes))

        print("Cleaned classes:")
        print(cleaned_classes)

        # Function to filter entities
        def filter_entities(example):
            filtered_entities = [
                entity for entity in example['entities']
                if 'entity_group' in entity and entity['entity_group'] in cleaned_classes
            ]
            return {"payload": example["payload"], "entities": filtered_entities}

        # Apply the filter to your dataset
        filtered_dataset = dataset.map(filter_entities, num_proc=num_proc)

        # Remove examples where entities are empty
        filtered_dataset = filtered_dataset.filter(
            lambda example: len(example['entities']) > 0, 
            num_proc=6
        )

        # Example to check if it worked
        print("\nOriginal first example:")
        print(dataset["train"][0]["entities"])
        print(dataset["train"][0]["payload"])
        
        print("\nFiltered first example:")
        print(filtered_dataset["train"][0]["entities"])
        print(filtered_dataset["train"][0]["payload"])

        # Check how many examples and entities were removed
        original_examples = len(dataset['train'])
        filtered_examples = len(filtered_dataset['train'])
        original_count = sum(len(example['entities']) for example in dataset['train'])
        filtered_count = sum(len(example['entities']) for example in filtered_dataset['train'])

        print(f"\nFor train set:")
        print(f"Removed {original_count - filtered_count} entities out of {original_count}")
        print(f"Removed {original_examples - filtered_examples} examples with empty entities out of {original_examples}")

        # Check how many examples and entities were removed
        original_examples = len(dataset['test'])
        filtered_examples = len(filtered_dataset['test'])
        original_count = sum(len(example['entities']) for example in dataset['test'])
        filtered_count = sum(len(example['entities']) for example in filtered_dataset['test'])

        print(f"\nFor test set:")
        print(f"Removed {original_count - filtered_count} entities out of {original_count}")
        print(f"Removed {original_examples - filtered_examples} examples with empty entities out of {original_examples}")

        dataset = filtered_dataset
        print("-" * 150)

    if args.concat_train_test:
        combined_dataset = datasets.concatenate_datasets([dataset["train"], dataset["test"]])
        dataset["train"] = combined_dataset
        dataset["test"] = combined_dataset

    # Remove meta column and rare labels
    print("\nRemove 'meta' column from the dataset, if there is any:")
    dataset = remove_meta_column(dataset)

    # Spojené štítky z obou částí
    def extract_entity_groups(example):
        return {"entity_groups": [entity["entity_group"] for entity in example["entities"] if "entity_group" in entity]}

    all_labels = []
    for ds in [dataset["train"], dataset["test"]]:
        mapped = ds.map(extract_entity_groups, batched=False, num_proc=num_proc)
        all_labels.extend([label for sublist in mapped["entity_groups"] for label in sublist])
    label_freq = Counter(all_labels)

    # Najdeme štítky s výskytem < 2
    rare_labels = {label for label, count in label_freq.items() if count < 2}
    print(f"Štítky s výskytem < 2 budou odstraněny: {rare_labels}")

    # Filtrování trénovacích a validačních vzorků pomocí lazy filter
    dataset["train"] = dataset["train"].filter(lambda example: not any("entity_group" in entity and entity["entity_group"] in rare_labels for entity in example["entities"]))
    dataset["test"] = dataset["test"].filter(lambda example: not any("entity_group" in entity and entity["entity_group"] in rare_labels for entity in example["entities"]))

    # ZNOVU spočítat unique labels po filtrování
    all_labels_filtered = []
    for ds in [dataset["train"], dataset["test"]]:
        mapped = ds.map(extract_entity_groups, batched=False, num_proc=num_proc)
        all_labels_filtered.extend([label for sublist in mapped["entity_groups"] for label in sublist])
    unique_entity_types = set(all_labels_filtered)


    # Convert the set to a list and format it
    entity_types = ["O"]
    for etype in unique_entity_types:
        entity_types.append(f"B-{etype}")
        entity_types.append(f"I-{etype}")
    entity_types.sort()
    label_list = entity_types

    train_labels = extract_all_labels(dataset["train"])
    test_labels = extract_all_labels(dataset["test"])
    missing_labels = set(test_labels) - set(train_labels)
    print(f"Chybějící štítky v trénovacích datech: {missing_labels}")

    # 3. Vyber záznamy z testu obsahující tyto chybějící štítky pomocí lazy filter
    test_samples_to_move_ds = dataset["test"].filter(lambda example: bool({entity["entity_group"] for entity in example["entities"] if "entity_group" in entity} & missing_labels))
    remaining_test_ds = dataset["test"].filter(lambda example: not bool({entity["entity_group"] for entity in example["entities"] if "entity_group" in entity} & missing_labels))

    print(f"Přesuneme {len(test_samples_to_move_ds)} záznamů z testu do trénovací množiny.")

    # 4. Nové datasety
    combined_train = datasets.concatenate_datasets([dataset["train"], test_samples_to_move_ds])
    new_test = remaining_test_ds
    new_dataset = DatasetDict({"train": combined_train, "test": new_test})

    train_label_counts = count_labels(new_dataset["train"])
    test_label_counts = count_labels(new_dataset["test"])

    print("-"  * 150)

    # Výpis výsledků
    print("\nPočet výskytů jednotlivých štítků v TRAIN:")
    for label, count in sorted(train_label_counts.items()):
        print(f"{label}: {count}")
    print("\nPočet výskytů jednotlivých štítků v TEST:")
    for label, count in sorted(test_label_counts.items()):
        print(f"{label}: {count}")

    print("-"  * 150)

    # %%
    train_set = new_dataset["train"]
    test_set = new_dataset["test"]

    # %%
    print("\nFinal dataset statistics after filtering:")
    print("\nTrain set size:", len(train_set))
    print("Test set size:", len(test_set))
    print("\nTrain set columns:", train_set.column_names)
    print("Test set columns:", test_set.column_names, "\n")

    # Create label2id / id2label
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    print("label_list:", label_list)
    print("label2id:", label2id)
    print("id2label:", id2label)

    num_of_labels = len(label_list)
    print(num_of_labels, "unique labels")
except Exception as e:
    _handle_fatal_exception(e, context_msg="data preprocessing")

print("-"  * 150)

# 4) Tokenizer / model preparation
try:
    print("\nSaving training samples.")
    train_samples = train_set.shuffle(seed=args.shuffle_seed).select(range(sample_range))
    train_results = []

    if args.augmentate:
        for samples in train_samples:
            texts = []
            all_entities =  []
            original_payload = samples["payload"]
            new_batch = [{"payload": samples["payload"], "entities": samples["entities"]}]
            new_batch = augmentator.augmentate(new_batch, count=1)
            if NORMALIZE_ENTITIES:
                new_batch = normalize_tokens_in_dataset({
                    "payload": [data["payload"] for data in new_batch],
                    "entities": [data["entities"] for data in new_batch]
                }, disable_placeholders=disable_placeholders)
            else:
                new_batch = {"payload": [data["payload"] for data in new_batch], "entities": [data["entities"] for data in new_batch]}
            for idx, payload in enumerate(new_batch["payload"]):
                texts.append(payload)
                all_entities.append(new_batch["entities"][idx])
            for idx, payload in enumerate(texts):
                formatted_predictions = []
                for entity in all_entities[idx]:
                    start, end = entity["start"], entity["end"]
                    word = payload[start:end]
                    formatted_predictions.append({
                        "entity": entity["entity_group"], "start": start, "end": end, "word": word
                    })
                train_results.append({"Original log": original_payload, "Augmented input": payload, "Entities": formatted_predictions})
    else:
        if NORMALIZE_ENTITIES:
            normalized_batch = normalize_tokens_in_dataset({"payload": train_samples["payload"], "entities": train_samples["entities"]}, disable_placeholders=disable_placeholders)
        else:
            normalized_batch = {"payload": train_samples["payload"], "entities": train_samples["entities"]}
        for idx, payload in enumerate(train_samples["payload"]):
            formatted_predictions = []
            for entity in normalized_batch["entities"][idx]:
                start, end = entity["start"], entity["end"]
                word = payload[start:end]
                formatted_predictions.append({"entity": entity["entity_group"], "start": start, "end": end, "word": word})
            train_results.append({"Input": payload, "Entities": formatted_predictions})

    if use_mlm_model:
        model_load_path = mlm_model_dir
        tokenizer_load_path = model_name.replace('|', '/')
    elif model_tokenizer_cache_path is not None:
        model_load_path = model_tokenizer_cache_path
        tokenizer_load_path = model_tokenizer_cache_path
    else:
        model_load_path = model_name.replace('|', '/')
        tokenizer_load_path = model_name.replace('|', '/')

    print("-"  * 150)
    print("\nModel load path:", model_load_path)
    print("Tokenizer load path:", tokenizer_load_path)

    max_length = args.max_length
    print("\nPreparing tokenizer with max_length:", max_length)
    if model_name == "lstmner":
        num_label_outputs = len(entity_types)
        tokenizer = LSTMNERTokenizer(tokenizer_name='t5-small', padding='max_length', truncation=True, max_length=max_length).get_tokenizer()
        lstmner_model_config = LSTMNERConfig(vocab_size=tokenizer.vocab_size, num_label_outputs=num_label_outputs)
        model = LSTMNERModel(lstmner_model_config)
    elif model_name == "roberta-base" or model_name == "allenai/longformer-base-4096" or model_name == "microsoft/deberta-base" or model_name=="kssteven/ibert-roberta-base" or model_name == "mnaylor/mega-base-wikitext" or model_name == "uw-madison/yoso-4096":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, add_prefix_space=True)
    elif model_name == "QDQBertModel":
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    elif "canine" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("google/canine-s", truncation=True, max_length=max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, truncation=True, model_max_length=max_length, vocab_size=args.vocab_size)

    print("\nUsing tokenizer:", tokenizer)

    print("\nOriginal tokenization example:")
    print(tokenizer.tokenize("src=192.168.45.66 dst=10.0.0.5"))

    if args.train_tokenizer:
        print("-"  * 150)
        print("\nTraining a new tokenizer with vocab size:", args.vocab_size)
        training_corpus = get_tokenizer_training_corpus(dataset)
        tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size=args.vocab_size, model_max_length=max_length) 
        
        print(tokenizer.tokenize("src=192.168.45.66 dst=10.0.0.5"))
        print("\nTrained tokenizer:", tokenizer)

    if args.make_custom_tokenizer:
        print("Creating custom tokenizer with vocab size:", args.vocab_size)
        tokenizer = IPTokenizer(tokenizer)

        print(tokenizer.tokenize("src=192.168.45.66 dst=10.0.0.5"))
        print("\nCustom tokenizer:", tokenizer)

    print()
    print("-"  * 150)

    print("\nExample payloads from the training set:")
    for i in range(sample_range):
        print(f"Sample {i}:", dataset["train"][i]["payload"])
    print("\n")

    # %%
    if model_name == "uw-madison/nystromformer-512":
        max_length = 510

    if ADD_SPECIAL_TOKENS:
        print("-"  * 150)
        processor = LogProcessor(disabled=disable_placeholders)
        tokens = processor.placeholders
        num_added = tokenizer.add_tokens(tokens)
        print(f"Added {num_added} special tokens to the tokenizer")
        verify_tokens(tokenizer, tokens)

    if model_name == "lstmner":
        is_encoder_decoder = False
        print("\nModel is encoder-decoder:", is_encoder_decoder)
    else:
        config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = config.is_encoder_decoder
        print("\nModel is encoder-decoder:", is_encoder_decoder)

    align_fn = partial(
        align_labels_with_tokens,
        tokenizer=tokenizer,
        label2id=label2id,
        model_name=model_name,
        max_length=max_length,
        is_encoder_decoder=is_encoder_decoder,
        augmentator=augmentator,
        tokenize_with_offsets=tokenize_with_offsets,
        add_payload_to_output=False,
        normalize_entities=NORMALIZE_ENTITIES,
        disable_placeholders=disable_placeholders if NORMALIZE_ENTITIES else None,
    )

    samples = dataset["train"].select(range(sample_range))
    processed_samples = samples.map(
        partial(align_fn, augment_data=False),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    print("-"  * 150)
    print("\nColumns in the processed samples:", processed_samples.column_names)

    for i in range(len(processed_samples)):
        print(f"Sample {i}:")
        print("Payload:", samples["payload"][i])
        print("Input IDs:", processed_samples["input_ids"][i])
        print("Tokens:", tokenizer.convert_ids_to_tokens(processed_samples["input_ids"][i]))
        print("Labels:", processed_samples[i]["labels"])
        # print("Offset Mapping:", processed_samples["offset_mapping"][i])
        # print("Attention Mask:", processed_samples["attention_mask"][i])
        if "payload" in processed_samples:
            print("Payload:", processed_samples["payload"][i])
        print("\n")

    print("-"  * 150)

    # %%
    if args.augmentate_valid:
        print("\nAligning labels with tokens for test_set with augmentation")
        dataset["test"].set_transform(partial(align_fn, augment_data=True))
        test_part = dataset["test"]
    else:
        print("\nAligning labels with tokens for test_set without augmentation")
        test_part = dataset["test"].map(
            partial(align_fn, augment_data=False),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True
        )
        test_part = test_part.remove_columns(["entities", "payload"])

    print("\nColumns in the test_part after aligning:", test_part.column_names)
    print(f"Test sample 0 - input_ids: {tokenizer.decode(test_part[0]['input_ids'], skip_special_tokens=True)}")

    if AUGMENTATE:
        print("\nAligning labels with tokens for train_set with augmentation")
        dataset["train"].set_transform(partial(align_fn, augment_data=True))
    else:
        print("\nAligning labels with tokens for train_set without augmentation")
        dataset["train"] = dataset["train"].map(
            partial(align_fn, augment_data=False), 
            batched=True, 
            num_proc=num_proc, 
            load_from_cache_file=True
        )

    print("\nColumns in the train_set after aligning:", dataset["train"].column_names, "\n")
    print("-"  * 150)

    # %%
    os.environ["WANDB_DISABLED"] = "true"

    # Definice modelu
    # if use_mlm_model:
    if model_name == "lstmner":
        train_batch_size = 256 # // 2
        eval_batch_size = train_batch_size # // 2
    # elif model_name == "google-bert/bert-base-cased":
    #     model = AutoModelForTokenClassification.from_pretrained(model_name, hidden_dropout_prob=0.1)
    elif model_name == "bhadresh-savani/electra-base-discriminator-finetuned-conll03-english":    
        model = AutoModelForTokenClassification.from_pretrained(model_load_path, num_labels=len(label_list), ignore_mismatched_sizes=True, local_files_only=True)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_load_path, num_labels=len(label_list), local_files_only=False)

    # Aktivace gradient checkpointingu pro velké modely
    if hasattr(model, "gradient_checkpointing_enable") and (model_name != "mnaylor/mega-base-wikitext" and model_name != "kssteven/ibert-roberta-base" and model_name != "lstmner"):
        model.gradient_checkpointing_enable()

    if model_name == "t5-small" or model_name == "google/t5-v1_1-base" or model_name == "google/flan-t5-small":
        model.model_parallel = False
        train_batch_size = 16 # // 2
        eval_batch_size = 1 # // 2
    elif model_name == "mnaylor/mega-base-wikitext":
        train_batch_size = 64 # // 2
        eval_batch_size = 1 # // 2

    if model_name == "facebook/xmod-base":
        model.set_default_language("en_XX")

    if args.train_tokenizer or args.make_custom_tokenizer:
        model.resize_token_embeddings(len(tokenizer))

    if ADD_SPECIAL_TOKENS:
        model.resize_token_embeddings(len(tokenizer))

    if DOWNLOAD_FOR_RETRAIN:
        print(f"Loading model from MLflow run ID: {DOWNLOAD_FOR_RETRAIN}")

        # Získání názvu runu
        try:
            run = mlflow.get_run(DOWNLOAD_FOR_RETRAIN)
            run_name = run.data.tags.get('mlflow.runName', 'Unnamed Run')
            print(f"\nRun name: {run_name}")
        except Exception as e:
            print(f"Could not retrieve run name: {e}")
            run_name = "Unnamed Run"
        
        # Define the MLflow model URI
        model_uri = f"runs:/{DOWNLOAD_FOR_RETRAIN}/model"

        # Download the model and tokenizer artifacts from MLflow
        local_model_path = mlflow.artifacts.download_artifacts(model_uri + "/model", dst_path=base_dir)
        print(f"\nModel downloaded to: {local_model_path}")

        local_tokenizer_path = mlflow.artifacts.download_artifacts(model_uri + "/components/tokenizer", dst_path=base_dir)
        print(f"\nTokenizer downloaded to: {local_tokenizer_path}")

        # Load the model using AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(
            local_model_path,
            num_labels=num_of_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        print(f"\nModel loaded from {local_model_path}")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, add_prefix_space=True)
        print(f"Tokenizer loaded from {local_tokenizer_path}")

        # 1. Freeze everything
        for param in model.base_model.parameters():
            param.requires_grad = False
        
        # 2. Unfreeze last N transformer layers + classifier
        N = args.num_layers_unfreeze  # number of last layers you want to unfreeze

        for layer in model.base_model.encoder.layer[-N:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # 3. Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # Creating datacollator with dynamic padding
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8 if args.use_fp16 or use_bf16 else None)
    
except Exception as e:
    _handle_fatal_exception(e, context_msg="tokenizer/model preparation")

print("\nMlflow run name:", run_name, "\n")

metrics_history = []
# konfigurovatelné
SEQEVAL_SAMPLE_SIZE = 500      # jak velký sample pro seqeval (entity-level). Sniž pro rychlost.
SAMPLE_PRED_K = 100            # kolik sekvencí uložit do sample preds/preds_files

def compute_metrics(eval_preds):
    t0 = time.time()
    step = int(getattr(trainer.state, "global_step", 0))

    preds, labels = eval_preds
    # preds může být tuple (logits, )
    if isinstance(preds, tuple):
        preds = preds[0]

    y_true_parts = []
    y_pred_parts = []

    for pred_seq, lab_seq in zip(iter_token_seqs(preds), iter_token_seqs(labels)):
        arr_pred = to_numpy(pred_seq)
        arr_lab  = to_numpy(lab_seq)
        mask = (arr_lab != -100)
        if mask.any():
            y_true_parts.append(arr_lab[mask].astype(np.int32, copy=False))
            y_pred_parts.append(arr_pred[mask].astype(np.int32, copy=False))
    if len(y_true_parts) == 0:
        # nic k vyhodnoceni
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    y_true = np.concatenate(y_true_parts, axis=0)
    y_pred = np.concatenate(y_pred_parts, axis=0)

    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    o_id = label2id.get("O", None)
    if o_id is not None:
        mask_no_o = (y_true != o_id)
        if mask_no_o.any():
            weighted_f1_no_O = float(f1_score(y_true[mask_no_o], y_pred[mask_no_o], average="weighted", zero_division=0))
        else:
            weighted_f1_no_O = 0.0
    else:
        weighted_f1_no_O = weighted_f1

    if CALC_PER_CLASS_F1:
        # labels for classification_report: all int ids that exist in id2label
        labels_ids = list(range(len(label_list)))
        report = classification_report(
            y_true, y_pred, labels=labels_ids, target_names=label_list, output_dict=True, zero_division=0
        )
        per_class_f1 = {lab: report[lab]['f1-score'] for lab in report if lab not in ['accuracy', 'macro avg', 'weighted avg']}
        # uložit per-class JSON (převod np int na python)
        with open(f"{eval_output_dir}/per_class_f1_step_{step}.json", "w") as f:
            json.dump(per_class_f1, f, indent=2)
    else:
        per_class_f1 = {}

    # --- 3) seqeval (entity-level) only on a sample to save time ---
    seqeval_metrics = {}
    try:
        sample_preds = []
        sample_labels = []
        
        cnt = 0
        for pred_seq, lab_seq in zip(iter_token_seqs(preds), iter_token_seqs(labels)):
            if cnt >= SEQEVAL_SAMPLE_SIZE:
                break
            pa = to_numpy(pred_seq)
            la = to_numpy(lab_seq)
            mask = (la != -100)
            if not mask.any():
                continue
            pred_ids = pa[mask].astype(int)
            lab_ids  = la[mask].astype(int)
            sample_preds.append([label_list[int(x)] for x in pred_ids])
            sample_labels.append([label_list[int(x)] for x in lab_ids])
            cnt += 1
        if len(sample_preds) > 0:
            seqeval_metrics = metric.compute(predictions=sample_preds, references=sample_labels, zero_division=0)
        else:
            seqeval_metrics = {"overall_precision": 0.0, "overall_recall": 0.0, "overall_f1": 0.0, "overall_accuracy": 0.0}
    except Exception:
        seqeval_metrics = {"overall_precision": 0.0, "overall_recall": 0.0, "overall_f1": 0.0, "overall_accuracy": 0.0}

    
    sample_k = SAMPLE_PRED_K
    small_labels = []
    small_preds = []
    k = 0
    for pred_seq, lab_seq in zip(iter_token_seqs(preds), iter_token_seqs(labels)):
        if k >= sample_k:
            break
        pa = to_numpy(pred_seq)
        la = to_numpy(lab_seq)
        mask = (la != -100)
        if not mask.any():
            continue
        small_labels.append([label_list[int(x)] for x in la[mask]])
        small_preds.append([label_list[int(x)] for x in pa[mask]])
        k += 1
    with open(f"{eval_output_dir}/predictions_step_{step}.json", "w") as f:
        json.dump({"labels": small_labels, "predictions": small_preds}, f)

    
    metrics_to_log = {
        "weighted_f1": weighted_f1,
        "weighted_f1_no_O": weighted_f1_no_O,
        "seqeval_overall_precision": float(seqeval_metrics.get("overall_precision", 0.0)),
        "seqeval_overall_recall": float(seqeval_metrics.get("overall_recall", 0.0)),
        "seqeval_overall_f1": float(seqeval_metrics.get("overall_f1", 0.0)),
        "seqeval_overall_accuracy": float(seqeval_metrics.get("overall_accuracy", 0.0)),
    }
    if CALC_PER_CLASS_F1:
        metrics_to_log.update({f"f1_{label}": f1 for label, f1 in per_class_f1.items()})

    metrics_entry = {"step": step}
    metrics_entry.update(metrics_to_log)
    metrics_history.append(convert_to_seializable(metrics_entry))

    if USE_MLFLOW:
        try:
            mlflow.log_metrics(metrics_to_log, step=step)
        except Exception as e:
            # Log error, but don't block training/eval flow
            print(f"Warning: mlflow.log_metrics failed: {e}")

    # print couple example sequences (small)
    for i in range(min(2, len(small_labels))):
        print(f"\n--- Example {i} ---")
        print("Gold labels:", small_labels[i])
        print("Predicted labels:", small_preds[i])

    # save historical metrics file (overwrites)
    with open(f"{base_dir}/metrics.json", "w") as f:
        json.dump(convert_to_seializable(metrics_history), f)

    torch.cuda.empty_cache()

    # combine results for Trainer
    result = {
        "precision": seqeval_metrics.get("overall_precision", 0.0),
        "recall": seqeval_metrics.get("overall_recall", 0.0),
        "f1": seqeval_metrics.get("overall_f1", weighted_f1),
        "accuracy": seqeval_metrics.get("overall_accuracy", 0.0),
        "weighted_f1": weighted_f1,
        "weighted_f1_no_O": weighted_f1_no_O
    }

    if CALC_PER_CLASS_F1:
        result.update({f"f1_{label}": f1 for label, f1 in per_class_f1.items()})

    elapsed = time.time() - t0
    print(f"\n[compute_metrics] done in {elapsed:.2f}s, examples flattened: {y_true.shape[0]}\n")
    return result

def eval_compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    small_labels = []
    small_preds = []
    k = 0
    for pred_seq, lab_seq in zip(iter_token_seqs(preds), iter_token_seqs(labels)):
        if k >= SAMPLE_PRED_K:
            break
        pa = to_numpy(pred_seq)
        la = to_numpy(lab_seq)
        mask = (la != -100)
        if not mask.any():
            continue
        small_labels.append([label_list[int(x)] for x in la[mask]])
        small_preds.append([label_list[int(x)] for x in pa[mask]])
        k += 1

    torch.cuda.empty_cache()
    return {"predictions": small_preds, "labels": small_labels}

print(upper_border)
print(inner)
print("┃{:^150}┃".format("STARTING TRAINING PROCESS"))
print(inner)
print(lower_border)
print("\n")

print("Model architecture for training:\n", model)
print("\n")
print("Tokenizer architecture for training:", tokenizer)

if USE_MLFLOW:
    hf_dataset = mlflow.data.huggingface_dataset.from_huggingface(
        train_set_for_logging,
        path=dataset_path,
        name=dataset_name,
    )
    
    mlflow.log_input(hf_dataset, context="training_dataset")

if USE_MLFLOW:
    report_to = ["mlflow"]
else:
    report_to = ["none"]

# Calculate steps per epoch
steps_per_epoch = math.ceil(len(dataset["train"]) / train_batch_size)

if args.val_2_times:
    # Evaluate at half of the total training steps
    eval_steps = math.ceil((steps_per_epoch * num_epoch) // 2)
    # Ensure we evaluate at least once
    eval_steps = max(eval_steps, 1)
else:
    # Evaluate every epoch (convert epoch frequency to steps)
    eval_steps = steps_per_epoch

# Make sure save_steps equals eval_steps to satisfy load_best_model_at_end requirement
save_steps = eval_steps

common_args = dict(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    weight_decay=args.weight_decay,
    logging_dir=logging_dir,
    eval_strategy="steps",
    eval_steps=eval_steps,
    logging_strategy="epoch",
    logging_steps=args.logging_steps,
    remove_unused_columns=not args.augmentate,
    eval_accumulation_steps=eval_accumulation_steps,
    eval_do_concat_batches=False if is_dataset_too_large else True, # Disable concatenation for large eval datasets
    save_strategy="steps",
    save_steps=save_steps,  # Explicitly use the same value as eval_steps
    metric_for_best_model=args.metric_for_best_model,
    greater_is_better=True if args.metric_for_best_model == "eval_f1" else False,
    load_best_model_at_end=True,
    report_to=report_to,
    fp16=args.use_fp16,
    bf16=use_bf16,
    dataloader_num_workers=1 if is_dataset_too_large else num_proc,  # Use more workers for larger datasets, but keep it low for small datasets to avoid overhead
)

if not DOWNLOAD_FOR_RETRAIN:
    specific_args = dict(
        num_train_epochs=num_epoch,
        learning_rate=5e-05 if not is_encoder_decoder else 2e-5,
        warmup_steps=args.warmup_steps,
        eval_on_start=True,
    )
else:
    specific_args = dict(
        num_train_epochs=args.retrain_epochs,
        learning_rate=1e-05 if not is_encoder_decoder else 2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_on_start=False,
    )

training_args = TrainingArguments(
    **common_args,
    **specific_args
)

# Save training arguments to a file
training_args_file = files_dir + "/training_args.bin"
with open(training_args_file, 'wb') as f:
    pickle.dump(training_args, f)
# Log .env file if it exists
env_file_path = ".env"  # Update this if your .env file is in a different location
if os.path.exists(env_file_path):
    mlflow.log_artifact(env_file_path, "model/.env")

if USE_MLFLOW:
    mlflow.log_artifact(training_args_file, artifact_path="model/model")

    train_file = f"{files_dir}/train_data.txt" 
    with open(train_file, 'w', encoding="utf-8") as f:
        for result in train_results:
            if args.augmentate:
                f.write(f"Original log: {result['Original log']}\n")
                f.write(f"Augmented log: {result['Augmented input']}\n")
            else:
                f.write(f"Input: {result['Input']}\n")
            f.write("Entities:\n")
            f.write(json.dumps(result["Entities"], indent=4, ensure_ascii=False) + "\n\n")
    mlflow.log_artifact(train_file, artifact_path="model/predictions")
    print("-"  * 150)
    print("\nTraining data logged to MLflow")

model.config.id2label = id2label
model.config.label2id = label2id

if NORMALIZE_ENTITIES:
    model.config.update({"_processor_disabled_placeholders": disable_placeholders})
    model.config.update({"_processor_enabled_placeholders": tokens})

if PREPROCESS_DATASET:
    model.config.update({"_preprocessor":
    {
        "to_lower": log_processor.to_lower,
        "normalize_ws": log_processor.normalize_ws,
        "remove_repeats": log_processor.remove_repeats,
        "strip_bracket_spaces": log_processor.strip_bracket_spaces,
        "drain": log_processor.drain
    }
})

print("-"  * 150)


if args.use_class_weight or model_name == "lstmner":
    if args.use_class_weight:
        all_labels = [label for example in dataset["train"] for label in example['labels']]
        label_counts = Counter(all_labels)
        total_labels = len(all_labels)
        num_classes = len(label_counts)

        class_weights = {label: total_labels / (num_classes * count) for label, count in label_counts.items()}
        print("label2id:", label2id)
        print("id2label:", id2label)
        print("class_weights:", class_weights)
        class_weights_list = [class_weights[label] for label in id2label]
    else:
        class_weights_list = None

    trainer = SectechTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=test_part,
        compute_metrics=compute_metrics, 
        class_weights=class_weights_list,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=test_part,
        compute_metrics=compute_metrics, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

try:
    trainer.train()
except Exception as e:
    _handle_fatal_exception(e, context_msg="training loop")


print("-"  * 150)
print("\nTraining completed with epochs: " + str(trainer.state.epoch))
if num_epoch != trainer.state.epoch:
    print("Note: Early stopping was triggered before reaching the maximum number of epochs.")

with open(f"{base_dir}/metrics.json", "w") as f:
    json.dump(metrics_history, f, indent=2)
mlflow.log_artifact(f"{base_dir}/metrics.json", artifact_path="model/eval_outputs")

if USE_MLFLOW:
    # Log the best model to MLflow
    best_chkpt = trainer.state.best_model_checkpoint
    print(f"\nLogging model from checkpoint {best_chkpt}")

    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        task="token-classification",
        save_format="safetensors",
        registered_model_name=None,
        save_pretrained=True,
    )
    
if not DO_NOT_EVALUATE:
    print()
    print(upper_border)
    print(inner)
    print("┃{:^150}┃".format("EVALUATING MODEL on validation dataset..."))
    print(inner)
    print(lower_border)
    print("\n")

    trainer.compute_metrics = eval_compute_metrics
    evaluation_results = trainer.evaluate()

    print()

    true_labels = evaluation_results["eval_labels"]
    predicted_labels = evaluation_results["eval_predictions"]

    true_labels_flat = [label for doc in true_labels for label in doc]
    predicted_labels_flat = [label for doc in predicted_labels for label in doc]

    print("-"  * 150)

    # Display the confusion matrix
    print("\nGenerating confusion matrix...")
    fig, ax = plt.subplots(figsize=(100, 100))
    cm = confusion_matrix(true_labels_flat, predicted_labels_flat)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_title("Confusion matrix")
    ax.set_ylabel("Actual label")
    ax.set_xlabel("Predicted label")

    # Save the figure in 4K resolution
    fig.savefig(os.path.join(validation_dir, "conf_m.png"))
    print(f"\nConfusion matrix saved to {os.path.join(validation_dir, 'conf_m.png')}")

    # Compute the classification report
    report = classification_report(
        true_labels_flat, predicted_labels_flat,
        zero_division=0
    )

    print("-"  * 150)

    # Print the classification report
    print("\nClassification Report:\n")
    print(report)
    print("-"  * 150)

    # Open the text file in write mode
    with open(os.path.join(validation_dir, 'classification_report.txt'), 'w') as f:
        # Write the classification report to the file
        f.write(report)
    
    # Získej pouze ty labely, které se opravdu vyskytují
    # Získej unikátní labely
    all_possible_labels = list(model.config.id2label.values())
    # unique_labels = sorted(set(true_labels_flat) | set(predicted_labels_flat))

    # Vytvoř matici záměn
    cm = confusion_matrix(true_labels_flat, predicted_labels_flat, labels=all_possible_labels)

    # DataFrame s popisy os
    df_cm = pd.DataFrame(cm, index=all_possible_labels, columns=all_possible_labels)
    df_cm.to_excel(os.path.join(validation_dir, "conf_m.xlsx"))

    # Normalize the confusion matrix
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        row_sums,
        where=row_sums != 0
    )

    # Convert the normalized confusion matrix to a DataFrame
    df_cm_normalized = pd.DataFrame(cm_normalized, index=all_possible_labels, columns=all_possible_labels)

    # Export the normalized DataFrame to an Excel file
    df_cm_normalized.to_excel(os.path.join(validation_dir, "conf_m_norm.xlsx"))

    print(f"Using existing run: {mlflow.active_run().info.run_id}")
    mlflow.log_artifact(validation_dir, artifact_path="model/")
    

if USE_MLFLOW:
    # Log an example of the test data to MLflow
    test_samples = test_set.shuffle(seed=args.shuffle_seed).select(range(sample_range))
    test_results = []

    if NORMALIZE_ENTITIES:
        normalized_batch = normalize_tokens_in_dataset(
            {"payload": test_samples["payload"], "entities": test_samples["entities"]},
            disable_placeholders=disable_placeholders
        )
        test_payloads = normalized_batch["payload"]
    else:
        test_payloads = test_samples["payload"]

    print("\nInitializing NER pipeline for predictions...")
    # Run prediction on a sample of the test set and log the results to MLflow
    nlp = pipeline(
        'ner', 
        model=model, tokenizer=tokenizer,
        device=0, batch_size=20, aggregation_strategy="simple"
    )
    
    print("NER pipeline initialized:", nlp)
    print("-"  * 150)
    
    print("\nRunning predictions on test samples...")
    for idx, payload in enumerate(test_payloads):
        ner_results = nlp(payload)
        formatted_predictions = [
            {
                "entity": entity["entity_group"], 
                "word": entity["word"] if is_encoder_decoder else payload[entity['start']:entity['end']],
                "score": float(round(entity["score"], 2)), 
                "start": entity["start"],
                "end": entity["end"]
            } 
            for entity in ner_results
        ]
        test_results.append({"Input": payload, "Predictions": formatted_predictions})
    print("Predictions completed.")
    print("-"  * 150)

    predictions_file = f"{files_dir}/ner_predictions.txt"
    with open(predictions_file, 'w', encoding="utf-8") as f:
        for result in test_results:
            f.write(f"Input: {result['Input']}\n")
            f.write("Predictions:\n")
            f.write(json.dumps(result["Predictions"], indent=4, ensure_ascii=False) + "\n\n")
    mlflow.log_artifact(predictions_file, artifact_path="model/predictions")
    
# %%
best_model_path = trainer.state.best_model_checkpoint
print("\nBest model path:", best_model_path)

# %%
if USE_MLFLOW:
    if os.path.exists(console_output_dir):
        mlflow.log_artifact(console_output_dir, artifact_path="model/console_output")
    
    if os.path.exists(eval_output_dir):
        mlflow.log_artifact(eval_output_dir, artifact_path="model/eval_outputs")

    print("\nModel and metrics logged to MLflow")
    mlflow.end_run(status="FINISHED")
    print("-"  * 150)
    print("\nRemoving checkpoint directory if exists...")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print("Checkpoint directory removed.")
    print("-"  * 150)

print(upper_border)
print(inner)
print("┃{:^150}┃".format("END OF TRAINING SCRIPT"))
print(inner)
print(lower_border)