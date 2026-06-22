from datasets import load_from_disk, DatasetDict
from src.training_functions import generate_dataset
from src.json_dataset_to_hugging_face import process_datasets
import os


def load_dataset(args):
    """Load dataset according to args and return (dataset, dataset_name).

    Supports synthetic generation, HF disk loads and JSON/folder processing.
    """
    dataset = None

    if args.generate_synthetic_train and args.generate_synthetic_valid:
        dataset = generate_dataset(n_samples=args.synthetic_train_samples, part="train")
        dataset["test"] = generate_dataset(n_samples=args.synthetic_valid_samples)

    elif args.generate_synthetic_train:
        dataset = generate_dataset(n_samples=args.synthetic_train_samples, part="train")
        dataset_from_disk = load_from_disk(args.dataset_path)
        dataset["test"] = dataset_from_disk["test"]

    elif args.generate_synthetic_valid:
        dataset = load_from_disk(args.dataset_path)
        dataset["test"] = generate_dataset(n_samples=args.synthetic_valid_samples)

    elif args.dataset_path:
        dataset = load_from_disk(args.dataset_path)

    elif args.json_dataset:
        dataset = process_datasets(
            args.json_dataset,
            single_file=True,
            mask_remove=True,
            meta_remove=True,
        )

    elif args.dataset_folder:
        dataset = process_datasets(
            args.dataset_folder,
            single_file=False,
            mask_remove=True,
            meta_remove=True,
        )

    else:
        raise ValueError("No dataset source provided. Set --dataset_path, --json_dataset, --dataset_folder or synthetic flags.")

    return dataset


__all__ = ["load_dataset"]
