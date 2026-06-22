#augmentator
from faker import Faker
import json
from sectech_log_augmentator.log_augmentator.custom_augmentator import CustomAugmentator
from sectech_log_augmentator.log_augmentator.custom_generator import CustomGenerator

def initialize_generator(seed):
    """Initializes and configures a CustomGenerator instance."""
    faker = Faker()
    Faker.seed(seed)
    generator = CustomGenerator(faker=faker)
    generator.create_code_function_providers()
    generator.create_custom_lists_providers()
    generator.test_generator()  # Run generator test within initialization
    return generator

def initialize_augmentator(seed = 0):
    """Initializes a CustomAugmentator instance using a configured generator."""
    generator = initialize_generator(seed)
    return CustomAugmentator(generator)

#simple generator
import random
import datetime
import ipaddress
import string
import re
from datasets import DatasetDict, Dataset

def generate_random_ipv4():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def generate_random_ipv6():
    return ":".join(''.join(random.choices("0123456789abcdef", k=4)) for _ in range(8))

def generate_random_ip():
    return generate_random_ipv4() if random.random() < 0.8 else generate_random_ipv6()

def generate_random_timestamp():
    formats = [
        "%b %d %H:%M:%S",               # Apr 21 21:03:11
        "%Y-%m-%d",                     # 2022-12-12
        "%H:%M:%S",                     # 01:07:33
        "%Y %b %d %H:%M:%S.%f",         # 2021 Mar 06 12:57:22.113432
        "%d/%m/%Y %H:%M:%S",            # 21/04/2023 21:03:11
        "%Y-%m-%dT%H:%M:%SZ"            # 2021-03-06T12:57:22Z
    ]
    dt = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 1_000_000))
    return dt.strftime(random.choice(formats))

def generate_random_port():
    return str(random.randint(1, 65535))

def generate_random_interface():
    types = [
        "eth", "lo", "wlan", "br", "bond", "tun", "tap",
        "gigabitEthernet", "fastEthernet", "serial", "atm", "ge", "xe"
    ]
    name = random.choice(types)
    if name in ["gigabitEthernet", "fastEthernet", "ge", "xe"]:
        index = f"{random.randint(0,3)}/{random.randint(0,3)}"
    else:
        index = f"{random.randint(0,10)}"
    return f"{name}{index}"

def generate_random_ip():
    """Vygeneruje náhodnou IPv4 nebo IPv6 adresu jako string."""
    if random.choice([True, False]):
        return str(ipaddress.IPv4Address(random.randint(0, (2**32) - 1)))
    else:
        return str(ipaddress.IPv6Address(random.getrandbits(128)))



def inject_random_ips(dataset, max_ips=3):
    """Vloží rovnoměrně rozdělené IP adresy do src=, dst= a device_ip pozic."""

    def insert_ips(example):
        text = example["payload"]
        entities = example["entities"]
        new_entities = entities.copy()
        offset = 0

        num_ips = random.randint(0, max_ips)
        if num_ips == 0:
            return example

        # Správně: vložit ihned po 'src=' a 'dst=' (před hodnotu)
        src_positions = [(m.start(1), "ip_src") for m in re.finditer(r'src=([^\s]*)', text)]
        dst_positions = [(m.start(1), "ip_dest") for m in re.finditer(r'dst=([^\s]*)', text)]
        dev_positions = [(m.start(), "device_ip") for m in re.finditer(r'(?<!\S)\S', text)]

        num_per_type = num_ips // 3
        remainder = num_ips % 3
        samples = []

        for positions, label, extra in zip(
            [src_positions, dst_positions, dev_positions],
            ["ip_src", "ip_dest", "device_ip"],
            [1 if i < remainder else 0 for i in range(3)]
        ):
            count = num_per_type + extra
            if positions:
                chosen = random.sample(positions, min(count, len(positions)))
                samples.extend(chosen)

        if not samples:
            return example

        for insert_at_raw, label in sorted(samples, key=lambda x: x[0]):
            ip = generate_random_ip()
            insertion = f"{ip}" if label in ("ip_src", "ip_dest") else f" {ip} "
            insert_at = insert_at_raw + offset

            text = text[:insert_at] + insertion + text[insert_at:]

            ip_start = insert_at if label in ("ip_src", "ip_dest") else insert_at + 1
            ip_end = ip_start + len(ip)

            new_entities.append({
                "start": ip_start,
                "end": ip_end,
                "entity_group": label
            })

            offset += len(insertion)


        return {
            "payload": text,
            "entities": sorted(new_entities, key=lambda x: x["start"])
        }

    dataset["train"] = dataset["train"].map(insert_ips)
    return dataset

def generate_random_tech_sentence(min_len=20, max_len=500):
    """Vytvoří náhodný technický text se simulovanými log prvky a surovými technickými větami."""
    
    prefixes = [
        "src=", "dst=", "interface=", "proto=", "event=", "log=", "dev=", "id=", 
        "status=", "mac=", "ver=", "fw=", "os=", "timestamp=", "msg=", "service="
    ]
    
    raw_fragments = [
        "Connection reset by peer",
        "Device unreachable on port 443",
        "Timeout occurred during handshake",
        "Segmentation fault in module kernel32.dll",
        "VPN tunnel established successfully",
        "Firewall dropped inbound packet",
        "Invalid credentials for user root",
        "Disk quota exceeded on /dev/sda1",
        "Service nginx restarted",
        "TLS handshake failed due to unknown CA",
        "Dropped packet due to malformed header",
        "Load average exceeded threshold",
        "Authentication failed for admin",
        "System uptime: 5 days 12:45:32",
        "ICMP echo request received from 192.168.0.1"
    ]
    
    words = []
    while True:
        if random.random() < 0.15:
            # Surový fragment vložíme s mezerami jako větu
            words.append(random.choice(raw_fragments))
        else:
            token = random.choice(prefixes) + ''.join(
                random.choices(string.ascii_letters + string.digits, k=random.randint(3, 10))
            )
            words.append(token)
        if len(' '.join(words)) >= max_len:
            break

    sentence = ' '.join(words)
    return sentence[:random.randint(min_len, max_len)]

import numpy as np
def convert_to_seializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_seializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_seializable(i) for i in obj]
    else:
        return obj

def generate_random_tech_sentence_with_entities(min_len=20, max_len=500):
    text = ""
    entities = []

    def add_field(label, value, use_label=True):
        nonlocal text
        entry = f"{label}={value}" if use_label else value
        # Ensure a space before label-less value if text is non-empty and doesn’t end with space
        if not use_label and text and not text.endswith(" "):
            text += " "
        start = len(text)
        text += entry + " "
        end = start + len(entry)
        entities.append({"start": start, "end": end, "entity_group": label})
        return entry

    # Random use of label=value or just value
    if random.random() < 0.5:
        add_field("timestamp", generate_random_timestamp(), random.random() < 0.5)
    if random.random() < 0.5:
        add_field("port_src", generate_random_port(), random.random() < 0.5)
    if random.random() < 0.5:
        add_field("port_dest", generate_random_port(), random.random() < 0.5)
    if random.random() < 0.5:
        add_field("interface_src", generate_random_interface(), random.random() < 0.5)
    if random.random() < 0.5:
        add_field("interface_dest", generate_random_interface(), random.random() < 0.5)
    if random.random() < 0.5:
        add_field("ip_src", generate_random_ip(), True)
    if random.random() < 0.5:
        add_field("ip_dest", generate_random_ip(), True)
    if random.random() < 0.5:
        add_field("device_ip", generate_random_ip(), random.random() < 0.5)

    # Random filler tokens
    prefixes = [
        "src=", "dst=", "proto=", "event=", "log=", "dev=", "id=", 
        "status=", "mac=", "ver=", "fw=", "os=", "msg=", "service="
    ]
    while len(text) < max_len:
        token = random.choice(prefixes) + ''.join(
            random.choices(string.ascii_letters + string.digits, k=random.randint(3, 10))
        )
        if text and not text.endswith(" "):
            text += " "
        text += token + " "

    return {"payload": text.strip(), "entities": sorted(entities, key=lambda x: x["start"])}

def generate_synthetic_entry():
    result = generate_random_tech_sentence_with_entities()
    # num_ips = random.randint(0, 3)
    entities = []
    # offset = 0
    # insertion_points = sorted(random.sample(range(len(base_text)), num_ips))

    return {
        "payload": result["payload"],
        "entities": result["entities"]
    }

def generate_dataset(n_samples=10000, part=None):
    data = [generate_synthetic_entry() for _ in range(n_samples)]
    if part is None:
        dataset = Dataset.from_list(data)
    elif part == "train":
        dataset = DatasetDict({
            "train": Dataset.from_list(data)
        })
    elif part == "test":
        dataset = DatasetDict({
            "test": Dataset.from_list(data)
        })
    return dataset

#ostatní
from collections import Counter
import datasets
def remove_meta_column(dataset):
    for split in dataset.keys():
        if "meta" in dataset[split].column_names:
            print(f"-- Removing 'meta' column from the {split} dataset")
            dataset[split] = dataset[split].remove_columns(["meta"])
        else:
            print(f"No 'meta' column in the {split} dataset")
    return dataset

# Pomocná funkce pro extrakci štítků
def extract_all_labels(dataset):
    labels = []
    for example in dataset:
        labels.extend([entity["entity_group"] for entity in example["entities"] if "entity_group" in entity])
    return labels

# Pomocná funkce pro filtrování vzorků, které neobsahují vzácné štítky
def filter_samples(dataset, rare_labels):
    return [
        example for example in dataset
        if all("entity_group" in entity and entity["entity_group"] not in rare_labels for entity in example["entities"])
    ]

# 1. Extrakce labelů
def extract_all_labels(dataset):
    return [entity["entity_group"] for example in dataset for entity in example["entities"] if "entity_group" in entity]



def count_labels(dataset):
    label_freq = Counter()
    for example in dataset:
        for entity in example["entities"]:
            if "entity_group" in entity:
                label_freq[entity["entity_group"]] += 1
    return label_freq

def get_tokenizer_training_corpus(dataset):
    # Concatenate train and test splits
    train_data = dataset["train"]
    test_data = dataset["test"]
    combined_data = datasets.concatenate_datasets([train_data, test_data])
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, len(combined_data), 1000):
        samples = combined_data[start_idx : start_idx + 1000]
        yield samples["payload"]

#custom tokenizer
# ---------- 1) regexy ----------

_IPv4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
_IPv6 = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
)

def _mask_ip(text: str) -> str:
    text = _IPv4.sub("<IP>", text)
    return _IPv6.sub("<IP>", text)

# ---------- 2) wrapper ----------
from transformers import AutoTokenizer
class IPTokenizer:
    """AutoTokenizer + regex náhrada IP → <IP> (1 token)."""
    def __init__(self, base_tokenizer):
        self.base = base_tokenizer
        self.base.add_special_tokens({"additional_special_tokens": ["<IP>"]})

    def _pre(self, x):  return _mask_ip(x)
    def __call__(self, text, *a, **k): return self.base(self._pre(text), *a, **k)
    def tokenize(self, text, *a, **k): return self.base.tokenize(self._pre(text), *a, **k)
    def __getattr__(self, x): return getattr(self.base, x)   # proxy ostatní metody
    def __len__(self): return len(self.base)


def tokenize_with_offsets(text, tokenizer, max_length=2048):
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    offsets, idx = [], 0
    for tok in tokens:
        if tok in tokenizer.all_special_tokens:
            offsets.append((0, 0))      # [CLS], [PAD]…
        else:
            offsets.append((idx, idx + 1))
            idx += 1                    # ByT5 ⇒ 1 token = 1 byte
    enc["offset_mapping"] = [offsets]
    return enc


###### preprocessing functions ######
from sectech_preprocess.preprocess import Preprocessor

def preprocess_dataset(dataset):
    """Aplikuje preprocessing na dataset pomocí Preprocessoru."""
    log_processor = Preprocessor()
    print("\n")
    print("\nPreprocessing dataset...")
    dataset["train"] = dataset["train"].map(lambda x: log_processor.process_single_item(x, update_entities=True), batched=False)
    dataset["test"] = dataset["test"].map(lambda x: log_processor.process_single_item(x, update_entities=True), batched=False)

    print("\nDataset AFTER preprocessing:\n")
    print(type(dataset))
    print(dataset)
    print("Train:")
    print(dataset["train"].column_names)
    for index in range(4):
        print(f"{index}.", dataset["train"][index]["payload"])
        print(" ", dataset["train"][index]["entities"], "\n")

    print("\nTest:")
    print(dataset["test"][0]["payload"])
    print(dataset["test"][0]["entities"])

    return dataset, log_processor

def make_drain_dataset(dataset):
    """Aplikuje draining na dataset pomocí Preprocessoru."""
    log_processor = Preprocessor()

    print("Draining dataset...")
    dataset["train"] = dataset["train"].map(lambda x: log_processor.drain_text(x), batched=False)
    dataset["test"] = dataset["test"].map(lambda x: log_processor.drain_text(x), batched=False)

    print("\nDataset AFTER draining:\n")

    print(type(dataset))
    print(dataset)
    print("Train:")
    print(dataset["train"].column_names)
    for index in range(4):
        print(f"{index}.", dataset["train"][index]["payload"])
        print(" ", dataset["train"][index]["entities"], "\n")

    print("\nTest:")
    print(dataset["test"][0]["payload"])
    print(dataset["test"][0]["entities"])

    return dataset

from sectech_log_processor.log_processor import LogProcessor
def normalize_tokens_in_dataset(batch, disable_placeholders=None):
    """Normalizuje tokeny v datech pomocí LogProcessoru a vrátí upravený batch."""
    log_processor = LogProcessor(disabled=disable_placeholders)

    payloads, entities, replacements = [], [], []
    for p, e in zip(batch["payload"], batch["entities"]):
        out = log_processor.apply_placeholders({
            "payload": p,
            "entities": e
        })
        payloads.append(out["payload"])
        entities.append(out["entities"])
        replacements.append(out["replacements"])
    return {
        "payload": payloads,
        "entities": entities, 
        "replacements": replacements
    }

def restore_tokens_in_predictions(predictions, replacements, payload):
    """Obnoví původní tokeny v predikcích pomocí uložených náhrad"""
    log_processor = LogProcessor()
    restored_predictions = log_processor.restore_placeholders(predictions, replacements, payload)
    return restored_predictions

def verify_tokens(tokenizer, tokens):
    """
    Ověří, že všechny dané tokeny jsou v tokenizeru reprezentovány jako jeden token.
    Vypíše jejich ID, dekódovanou podobu a počet subtokenů.
    """
    for t in tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        print(f"Token: {t}")
        print(f"  IDs: {ids}")
        print(f"  Decoded: {decoded}")
        print(f"  Počet subtokenů: {len(ids)} {'✅ OK' if len(ids)==1 and decoded==t else '❌ CHYBA'}")
        print("-"  * 50)

import mlflow
from transformers import AutoModelForTokenClassification, AutoTokenizer

def just_run_evaluate(run_id, model_dir):
    """
    Načte model a tokenizer z MLflow podle run_id a vrátí je spolu s label2id a id2label.
    """
    print(f"Loading model from MLflow run ID: {run_id}")
    
    # Define the MLflow model URI
    model_uri = f"runs:/{run_id}/model"

    # Download the model and tokenizer artifacts from MLflow
    local_model_path = mlflow.artifacts.download_artifacts(model_uri + "/model", dst_path=model_dir)
    print(f"Model downloaded to: {local_model_path}")

    local_tokenizer_path = mlflow.artifacts.download_artifacts(model_uri + "/components/tokenizer", dst_path=model_dir)
    print(f"Tokenizer downloaded to: {local_tokenizer_path}")
    # Load the model using AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(local_model_path)
    print(f"Model loaded from {local_model_path}")

    # Load the tokenizer using AutoTokenizer and check if it supports `add_prefix_space`
    tokenizer_class = AutoTokenizer.from_pretrained(local_tokenizer_path).__class__.__name__
    print(f"Tokenizer class: {tokenizer_class}")

    tokenizer_without_prefix_space = [ 
        "T5Tokenizer", "T5TokenizerFast", "MT5Tokenizer", "MT5TokenizerFast",
        "ByT5Tokenizer", "ByT5TokenizerFast"
    ] 

    print("Tokenizer path:", local_tokenizer_path)
    print("Type:", type(local_tokenizer_path))

    # Load the tokenizer
    if tokenizer_class in tokenizer_without_prefix_space:
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, use_fast=True, add_prefix_space=False)
        print(f"Tokenizer {tokenizer_class} does not support `add_prefix_space`. Loaded without it.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, add_prefix_space=True)
        print(f"Tokenizer {tokenizer_class} supports `add_prefix_space=True`. Loaded with it.")

    label2id = model.config.label2id
    id2label = model.config.id2label

    return model, tokenizer, label2id, id2label

def align_labels_with_tokens(
    batch, 
    tokenizer, 
    label2id, 
    model_name, 
    max_length, 
    is_encoder_decoder, 
    augmentator=None, 
    tokenize_with_offsets=None,
    augment_data=False, 
    add_payload_to_output=False,
    normalize_entities=False,
    disable_placeholders=None,
):  
    """
    Zarovná štítky entit s tokeny pomocí tokenizeru.
    Podporuje augmentaci dat a normalizaci entit před tokenizací.
    """
    if augment_data and augmentator is not None:
        texts = []
        all_entities = []
        for batch_index in range(len(batch["payload"])):
            new_batch = [{"payload": batch["payload"][batch_index], "entities": batch["entities"][batch_index]}]
            new_batch = augmentator.augmentate(new_batch, count=1)
            
            if normalize_entities:
                new_batch = normalize_tokens_in_dataset(
                    {
                        "payload": [data["payload"] for data in new_batch],
                        "entities": [data["entities"] for data in new_batch]
                    },
                    disable_placeholders=disable_placeholders
                )
            else:
                new_batch = {
                    "payload": [data["payload"] for data in new_batch],
                    "entities": [data["entities"] for data in new_batch]
                }

            texts.extend(new_batch["payload"])
            all_entities.extend(new_batch["entities"])
    else:
        if normalize_entities:
            normalized_batch = normalize_tokens_in_dataset(
                {"payload": batch["payload"], "entities": batch["entities"]}, 
                disable_placeholders=disable_placeholders
            )
            texts = normalized_batch["payload"]
            all_entities = normalized_batch["entities"]
        else:
            texts = batch["payload"]
            all_entities = batch["entities"]

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for text, entities in zip(texts, all_entities):
        if "by" in model_name or "canine" in model_name:
            tokenized_input = tokenize_with_offsets(text, tokenizer, max_length=max_length)
            offset_mapping = tokenized_input["offset_mapping"][0]
        else:
            tokenized_input = tokenizer(
                text, 
                return_offsets_mapping=True, 
                return_tensors="pt", 
                padding=False, 
                truncation=True, 
                max_length=max_length
            )
            offset_mapping = tokenized_input["offset_mapping"][0].tolist()

        input_ids = tokenized_input["input_ids"][0]
        attention_mask = tokenized_input["attention_mask"][0].tolist()

        if not is_encoder_decoder:
            if "by" in model_name or "canine" in model_name:
                word_ids = [
                    None if (s == 0 and e == 0) else i
                    for i, (s, e) in enumerate(offset_mapping)
                ]
            else:
                word_ids = tokenized_input.word_ids(batch_index=0)

            labels = []
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    labels.append(-100)
                    continue

                token_start, token_end = offset_mapping[idx]
                assigned = False
                for entity in entities:
                    entity_start = entity["start"]
                    entity_end = entity["end"]
                    entity_label = entity["entity_group"]

                    if token_start == entity_start:
                        labels.append(label2id[f"B-{entity_label}"])
                        assigned = True
                        break
                    elif entity_start < token_start < entity_end:
                        labels.append(label2id[f"I-{entity_label}"])
                        assigned = True
                        break
                if not assigned:
                    labels.append(label2id["O"])
        else:
            labels = []
            for idx, (offset_start, offset_end) in enumerate(offset_mapping):
                token_id = input_ids[idx].item()
                token_str = tokenizer.convert_ids_to_tokens(token_id)

                if token_str in tokenizer.all_special_tokens:
                    labels.append(-100)
                    continue

                assigned = False
                for entity in entities:
                    start = entity["start"]
                    end = entity["end"]
                    label = entity["entity_group"]

                    if offset_start == start:
                        labels.append(label2id[f"B-{label}"])
                        assigned = True
                        break
                    elif start < offset_start < end:
                        labels.append(label2id[f"I-{label}"])
                        assigned = True
                        break
                if not assigned:
                    labels.append(label2id["O"])

        input_ids_list.append(input_ids.tolist())
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
        **({"payload": texts} if add_payload_to_output else {})
    }

def load_classes_from_json(path):
    """
    Načte třídy z JSON souboru s metaklíči.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        result = []
        for group in data:
            result.append(group["id"])  # add first-level id
            result.extend([m["name"] for m in group.get("metakeys", [])])  # add all metakey names

    return result
   
def remove_classes_from_json(path, classes_to_remove):
    """
    Vrátí seznam tříd (group id a metakey jmen), které zůstanou po odstranění
    zadaných tříd. NEZMĚNÍ původní JSON soubor.
    - `path`: cesta ke `categories.json`
    - `classes_to_remove`: iterovatelný seznam jmen skupin nebo metaklíčů, které
      se mají odstranit (např. ["other"]).
    Vrací: list[string] obsahující `group id` a `metakey` jména, které mají zůstat.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not classes_to_remove:
        # nothing to remove -> return all ids and metakey names
        result = []
        for group in data:
            result.append(group.get("id"))
            result.extend([m.get("name") for m in group.get("metakeys", [])])
        return result

    # Always ensure the 'other' class is removed as requested by user
    classes_to_remove_set = set(classes_to_remove) if classes_to_remove else set()

    kept = []
    for group in data:
        group_id = group.get("id")
        if group_id in classes_to_remove_set:
            # skip entire group
            continue
        # keep the group id itself
        kept.append(group_id)
        # keep only metakeys whose name is not in classes_to_remove
        for m in group.get("metakeys", []):
            name = m.get("name")
            if name not in classes_to_remove_set:
                kept.append(name)

    # Remove duplicates while preserving order
    seen = set()
    kept_ordered = []
    for x in kept:
        if x not in seen:
            kept_ordered.append(x)
            seen.add(x)

    return kept_ordered

# Create function that will search experiment_id by experiment_name
def get_experiment_id_by_name(run_name: str, client) -> str:
    """
    Vrátí experiment_id pro daný experiment_name.
    """
    runs = client.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        filter_string=f"name = '{run_name}'"
    )
    if runs:
        return runs[0].experiment_id
    else:
        print(f"No experiment found with name '{run_name}'")
        return None


def get_experiment_id(args, client):
    """
    Retrieve the experiment ID based on provided arguments.
    Prioritizes experiment_name over model_experiment_id.
    """
    if args.experiment_name:
        try:
            experiment_id = get_experiment_id_by_name(args.experiment_name, client)
            if experiment_id is None:
                raise ValueError(f"Experiment with name '{args.experiment_name}' not found.")
            print(f"Using experiment ID '{experiment_id}' for name '{args.experiment_name}'")
            return experiment_id
        except ValueError as e:
            print(f"Error retrieving experiment by name: {e}")
            raise
    elif args.model_experiment_id:
        print(f"Using provided experiment ID '{args.model_experiment_id}'")
        return args.model_experiment_id
    else:
        raise ValueError("You must provide either --experiment_name or --model_experiment_id")

from typing import Optional
# Create function that will search run_id by run_name and experiment_id
def get_run_id_by_name(
        experiment_id: str, 
        run_name: Optional[str] = None,
        tag_key: Optional[str] = None,
        tag_value: Optional[str] = None,
        client = None) -> Optional[str]:
    """
    Vrátí run_id pro daný run_name v rámci experiment_id.
    """
    filters = []
    if run_name is not None:
        filters.append(f"attributes.run_name = '{run_name}'")
    
    if tag_key is not None and tag_value is not None:
        filters.append(f"tags.\"{tag_key}\" = \"{tag_value}\"")
    filter_string = " AND ".join(filters)

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string
    )

    if runs:
        return runs[0].info.run_id
    else:
        print(f"No run found with name '{run_name}' in experiment ID '{experiment_id}'")
        return None

        
def get_run_id(args, experiment_id, tag_key, tag_value, client):
    """
    Retrieve the run ID based on provided arguments.
    Prioritizes run_name over model_run_id.
    """
    if args.run_name:
        try:
            run_id = get_run_id_by_name(
                experiment_id,
                args.run_name,
                tag_key=tag_key,
                tag_value=tag_value, 
                client=client
            )
            if run_id is None:
                raise ValueError(f"Run with name '{args.run_name}' in experiment '{experiment_id}' not found.")
            print(f"Using run ID '{run_id}' for name '{args.run_name}' in experiment '{experiment_id}'")
            return run_id
        except ValueError as e:
            print(f"Error retrieving run by name: {e}")
            raise
    elif args.model_run_id:
        print(f"Using provided run ID '{args.model_run_id}'")
        return args.model_run_id
    else:
        raise ValueError("You must provide either --run_name or --model_run_id")


# Funkce pro předzpracování logits pro metriky (výběr nejpravděpodobnějších štítků)
def preprocess_logits_for_metrics(logits, labels):
    # logits: (batch, seq_len, num_labels) nebo tuple; vezmeme první
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def to_numpy(x):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def iter_token_seqs(x):
    """
    Yields 1D sequences (L,) for each sample.
    Supports:
      - np/tensor: (N,L) or (N,L,C)
      - list/tuple of batches: [(B,L), ...] or [(B,L,C), ...]
    """
    import numpy as np

    def _yield_from_arr(arr):
        arr = to_numpy(arr)
        if arr.ndim == 3:           # logits -> pred_ids
            arr = arr.argmax(axis=-1)
        if arr.ndim == 2:           # (N,L)
            for row in arr:
                yield row
        elif arr.ndim == 1:         # (L,)
            yield arr
        else:
            raise ValueError(f"Unexpected shape in metrics: {arr.shape}")

    if isinstance(x, (list, tuple)):
        for batch in x:
            yield from _yield_from_arr(batch)
    else:
        yield from _yield_from_arr(x)