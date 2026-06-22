import argparse
import multiprocessing
import os


def parse_args():
    """Parse command-line arguments and set offline env if requested.
    Returns argparse.Namespace
    """
    parser = argparse.ArgumentParser()

    # Important arguments
    parser.add_argument('--model_name',                     type=str, default="jackaduma/SecBERT", help="Name of the model to be trained")
    parser.add_argument('--dataset_path',                   type=str, help="Path to the dataset in Hugging Face format")
    parser.add_argument('--json_dataset',                   type=str, help="Path to the single JSON format dataset")
    parser.add_argument('--dataset_folder',                 type=str, help="Path to the multiple JSON format datasets in a folder")
    parser.add_argument('--base_dir',                       type=str, required=True, help="Path to store all the output files.")
    parser.add_argument('--train_batch_size',               type=int, default=32, help="Training batch size")
    parser.add_argument('--eval_batch_size',                type=int, default=8, help="Evaluation batch size")
    parser.add_argument('--num_epochs',                     type=int, default=40, help="Number of epochs to train the model")

    # Mlflow arguments
    parser.add_argument('--experiment_id',                  type=str, default=933010621898514002, help="MLflow experiment ID to use for logging")
    parser.add_argument('--experiment_name',                type=str, default=None, help="MLflow experiment name to use for logging.")
    parser.add_argument('--run_name',                       type=str, default=None, help="Name of the MLflow run")
    parser.add_argument('--tag_key',                        type=str, default="model_type", help="Tag key to filter runs by")
    parser.add_argument('--tag_value',                      type=str, default="tuned", help="Tag value to filter runs by")

    # String arguments
    parser.add_argument('--model_tokenizer_cache_path',         type=str, required=False)
    parser.add_argument('--retrain_epochs',                     type=int, default=3, help="Number of epochs to retrain the model")

    parser.add_argument('--filter_classes', nargs='+',          type=str, help='List of class names to be loaded. --filter_classes ip_src ip_dst mac_dst')
    parser.add_argument('--path_filter_classes',                type=str, default='categories.json', help='Path to a json file containing class names to be kept, anything else will be filtered out.')
    parser.add_argument('--remove_class_from_json', nargs='+',  type=str, default=['other'], action='extend', help='List of class names to be deleted from categories.json. --filter_classes other, ip_src etc')

    parser.add_argument('--vocab_size',                         type=int, default=40000, help="Vocabulary size for the tokenizer")
    parser.add_argument('--download_for_retrain',               type=str, help="Download the model for retraining, if it is not available locally")

    # Boolean arguments
    parser.add_argument('--use_mlm_model',                      action='store_true', help="Use a pre-trained MLM model for token classification")
    parser.add_argument('--augmentate',                         action='store_true', help="Use data augmentation techniques during training")
    parser.add_argument('--augmentate_valid',                   action='store_true', help="Use data augmentation techniques during validation")
    parser.add_argument("--disable_mlflow",                     action="store_true", help="Disable MLflow tracking")
    parser.add_argument('--concat_train_test',                  action='store_true', help="Concatenate train and test datasets into one dataset for training")
    parser.add_argument('--use_fp16',                           action='store_true', help="Use mixed precision training with FP16")
    parser.add_argument('--use_class_weight',                   action='store_true', help="Use class weights for training")
    parser.add_argument('--no_calc_per_class_f1',               action='store_true', help="Do not calculate per-class F1 score during evaluation")
    parser.add_argument('--do_not_evaluate',                    action='store_true', help="Do not run evaluation after training")
    parser.add_argument('--val_2_times',                        action='store_true', help="Run validation twice during training, instead of more frequent evaluation")
    parser.add_argument('--generate_synthetic_train',           action='store_true', help="Generate synthetic training data")
    parser.add_argument('--generate_synthetic_valid',           action='store_true', help="Generate synthetic validation data")
    parser.add_argument('--train_tokenizer',                    action='store_true', help="Train a new tokenizer instead of using the pre-trained one")
    parser.add_argument('--make_custom_tokenizer',              action='store_true', help="Make a custom tokenizer instead of using the pre-trained one")
    parser.add_argument('--preprocess',                         action='store_true', help="Preprocess the dataset before training")
    parser.add_argument('--drain',                              action='store_true', help="Drain the dataset, i.e., remove entity values and replace them with <*>")
    parser.add_argument('--add_special_tokens_and_normalize',   action='store_true', help="Add special tokens to the tokenizer and normalize the dataset -- remove entity values and replace them with tokens (i.e. [IPV4])")

    # Others
    parser.add_argument('--random_seed',                        type=int, default=2024, help="Random seed for augmentation and reproducibility")
    parser.add_argument('--tracking_uri',                       type=str, default="http://mlflow:5000/", help="MLflow tracking server URI")
    parser.add_argument('--synthetic_train_samples',            type=int, default=10000, help="Number of synthetic training samples to generate")
    parser.add_argument('--synthetic_valid_samples',            type=int, default=2000, help="Number of synthetic validation samples to generate")
    parser.add_argument('--shuffle_seed',                       type=int, default=42, help="Seed for shuffling training samples")
    parser.add_argument('--max_length',                         type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument('--num_layers_unfreeze',                type=int, default=6, help="Number of last layers to unfreeze during retraining")
    parser.add_argument('--weight_decay',                       type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument('--warmup_steps',                       type=int, default=1000, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument('--logging_steps',                      type=int, default=20, help="Steps between logging")
    parser.add_argument('--eval_accumulation_steps',            type=int, default=16, help="Accumulation steps for evaluation")
    parser.add_argument('--metric_for_best_model',              type=str, default="eval_f1", help="Metric to monitor for best model")
    parser.add_argument('--greater_is_better',                  action='store_true', default=True, help="Whether higher metric values are better")
    parser.add_argument('--dataloader_num_workers',             type=int, default=multiprocessing.cpu_count(), help="Number of workers for data loading")
    parser.add_argument('--offline', action='store_true', help="Run in offline mode, without internet access")

    args = parser.parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_EVALUATE_OFFLINE"] = "1"

    return args


__all__ = ["parse_args"]
