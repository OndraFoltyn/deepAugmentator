#!/bin/bash

###############################
# Automated training wrapper
# This is a copy of train.sh modified to iterate
# WHITELIST_VALUES from 1 to 14 and WHOLE_LOG_WHITELIST_VALUES from 1 to 11 automatically.
###############################

# Base directory where all training outputs (models, logs) will be saved
base_dir="runs/JISA_experiments"

# Path to the root directory containing all datasets (each subfolder should be a dataset)
dataset_root="/share/dataset_for_JISA/sample_logs.json"     # "/8TB/personal_space/datasets/golden_dataset"

# Array of model names to train; you can add multiple models here
models=("jackaduma/SecBERT") # ("google/flan-t5-small")     

# Experiment identification (used for MLflow tracking or internal logging)
experiment_id=979961448656578544

# Create custom run name (optional)
run_name="Synthetic_UpperCase_part_0.6_model"

# Number of training epochs; can also be overridden via train.py arg --num_epochs
num_epochs=20

# Training batch size; can also be overridden via train.py arg --train_batch_size
train_batch_size=32

# Default augmentation parameters - multiple values allowed for grid search
# Changed: iterate from 1 to 14 for entity, 1 to 11 for whole automatically
WHITELIST_VALUES=($(seq 1 14))
PART_VALUES=("0.3")

WHOLE_LOG_WHITELIST_VALUES=($(seq 1 11))


# Path to environment file used for augmentation parameters
ENV_FILE=".env"

# Boolean flags: 1 = True (flag active), 0 = False (flag inactive)
USE_FP16=1
ADD_SPECIAL_TOKENS_AND_NORMALIZE=0
PREPROCESS=0
TRAIN_TOKENIZER=1
AUGMENTATE=1
AUGMENTATE_VALID=0
DISABLE_MLFLOW=0
CONCAT_TRAIN_TEST=0
USE_CLASS_WEIGHT=0
NO_CALC_PER_CLASS_F1=0
DO_NOT_EVALUATE=0
VAL_2_TIMES=0
GENERATE_SYNTHETIC_TRAIN=0
GENERATE_SYNTHETIC_VALID=0
MAKE_CUSTOM_TOKENIZER=0
DRAIN=0
USE_MLM_MODEL=0
DOWNLOAD_FOR_RETRAIN=0
PATH_FILTER_CATEGORIES=1

# Run ID for validation (typically unique per training run)
run_id="4579cabee2e6411db88490fb1158e71e"

# Number of epochs to retrain the model
retrain_epochs=3

# Path to categories file
categories_path="categories.json"

# Set vocabulary size
vocab_size=40000

# New variables for additional arguments (with defaults matching train.py)
random_seed=2024                                    # Seed for reproducibility
mlflow_tracking_uri="http://192.168.40.2:5000/"     # MLflow tracking server URI
synthetic_train_samples=10000                       # Number of synthetic samples to generate for training
synthetic_valid_samples=2000                        # Number of synthetic samples to generate for validation
shuffle_seed=42                                     # Seed for shuffling datatets
max_length=512                                      # Maximum sequence length for tokenization
num_layers_unfreeze=6                               # For retraining last N layers
weight_decay=0.01                                   # Weight decay for optimizer
logging_steps=20                                    # Logging steps during training
eval_accumulation_steps=16                          # Evaluation accumulation steps
metric_for_best_model="eval_f1"                     # Metric to monitor for best model
greater_is_better=1                                 # 1 = True, 0 = False (since it's a bool flag)
dataloader_num_workers=$(nproc)                     # Use nproc to get CPU count, matching multiprocessing.cpu_count()

###############################
# Internal functions (no need to change)
###############################

# Update .env file with current augmentation parameters for the training run
update_env_file() {
    local whitelist="$1"
    local aug_part="$2"
    local whole_whitelist="$3"
    
    if [ -f "$ENV_FILE" ]; then
        # Update AUGMENTATION_WHITELIST
        if grep -q "^AUGMENTATION_WHITELIST[[:space:]]*=[[:space:]]*" "$ENV_FILE"; then
            sed -i 's/^AUGMENTATION_WHITELIST[[:space:]]*=[[:space:]]*.*$/AUGMENTATION_WHITELIST = '"$whitelist"'/' "$ENV_FILE"
        else
            echo "AUGMENTATION_WHITELIST = $whitelist" >> "$ENV_FILE"
        fi
        
        # Update AUGMENTATION_PART
        if grep -q "^AUGMENTATION_PART[[:space:]]*=[[:space:]]*" "$ENV_FILE"; then
            sed -i 's/^AUGMENTATION_PART[[:space:]]*=[[:space:]]*.*$/AUGMENTATION_PART = '"$aug_part"'/' "$ENV_FILE"
        else
            echo "AUGMENTATION_PART = $aug_part" >> "$ENV_FILE"
        fi
        
        # Update WHOLE_LOG_AUGMENTATION_WHITELIST
        if grep -q "^WHOLE_LOG_AUGMENTATION_WHITELIST[[:space:]]*=[[:space:]]*" "$ENV_FILE"; then
            sed -i 's/^WHOLE_LOG_AUGMENTATION_WHITELIST[[:space:]]*=[[:space:]]*.*$/WHOLE_LOG_AUGMENTATION_WHITELIST = '"$whole_whitelist"'/' "$ENV_FILE"
        else
            echo "WHOLE_LOG_AUGMENTATION_WHITELIST = $whole_whitelist" >> "$ENV_FILE"
        fi
    else
        echo "Warning: .env file not found at $ENV_FILE"
        echo "Creating new .env file"
        echo "AUGMENTATION_WHITELIST = $whitelist" > "$ENV_FILE"
        echo "AUGMENTATION_PART = $aug_part" >> "$ENV_FILE"
        echo "WHOLE_LOG_AUGMENTATION_WHITELIST = $whole_whitelist" >> "$ENV_FILE"
    fi
}

add_flag() {
  local flag_name=$1
  local flag_value=$2
  if [ "$flag_value" -eq 1 ]; then
    echo "$flag_name"
  else
    echo ""
  fi
}

get_env_value() {
    local key="$1"
    if [ -f "$ENV_FILE" ]; then
        grep "^$key[[:space:]]*=[[:space:]]*" "$ENV_FILE" | sed 's/.*= *//' | tr -d ' '
    else
        echo ""
    fi
}

# Map numeric augmentation whitelist code to human-friendly name
get_whitelist_name() {
        case "$1" in
                1) echo "to_upper_case" ;;
                2) echo "random_swap_words" ;;
                3) echo "random_case" ;;
                4) echo "remove_stop_words" ;;
                5) echo "to_lower_case" ;;
                6) echo "word_dropout" ;;
                7) echo "ocr_augmentation" ;;
                8) echo "random_deletion" ;;
                9) echo "random_insertion" ;;
             10) echo "random_swap" ;;
             11) echo "keyboard_augmentation" ;;
             12) echo "character_substitution" ;;
             13) echo "modify_punctuation" ;;
             14) echo "blank_noising" ;;
                *) echo "whitelist_$1" ;;
        esac
}

# Map numeric whole-log augmentation whitelist code to human-friendly name
get_whole_whitelist_name() {
        case "$1" in
                1) echo "replace_spaces_with_underscore" ;;
                2) echo "replace_spaces_with_dash" ;;
                3) echo "wrap_text" ;;
                4) echo "normalize_unicode" ;;
                5) echo "replace_spaces_with_double_underscore" ;;
                6) echo "replace_spaces_with_double_dash" ;;
                7) echo "replace_spaces_with_double_punctuations" ;;
                8) echo "remove_punctuation_symbols" ;;
                9) echo "replace_spaces_with_single_punctuation" ;;
             10) echo "to_lower_case" ;;
             11) echo "reduce_multiple_spaces_to_single_space" ;;
                *) echo "whole_whitelist_$1" ;;
        esac
}

run_training() {
    local whitelist="$1"
    local aug_part="$2"
    local whole_whitelist="$3"

    echo "#################################################"
    echo "Training model: $model"
    echo "Training on dataset: $dataset_folder"

    # Read ENABLE_SYNTHETIC_DATA
    local ENABLE_SYNTHETIC=$(get_env_value "ENABLE_SYNTHETIC_DATA")
    local prefix=""
    if [ "$ENABLE_SYNTHETIC" = "True" ]; then
        prefix="synthetic_"
    fi

    # Determine descriptive run name based on whitelists
    local WHITELIST_NAME=""
    local WHOLE_WHITELIST_NAME=""
    if [ -n "$whitelist" ]; then
        WHITELIST_NAME=$(get_whitelist_name "$whitelist")
    fi
    if [ -n "$whole_whitelist" ]; then
        WHOLE_WHITELIST_NAME=$(get_whole_whitelist_name "$whole_whitelist")
    fi

    local current_run_name
    if [ -n "$whitelist" ] && [ -n "$whole_whitelist" ]; then
        current_run_name="${prefix}${WHITELIST_NAME}_${WHOLE_WHITELIST_NAME}"
    elif [ -n "$whitelist" ]; then
        current_run_name="${prefix}${WHITELIST_NAME}"
    elif [ -n "$whole_whitelist" ]; then
        current_run_name="${prefix}whole_${WHOLE_WHITELIST_NAME}"
    else
        current_run_name="${prefix}no_aug"
    fi

    # Update the .env file with current parameters
    update_env_file "$whitelist" "$aug_part" "$whole_whitelist"

    local model_train_base_dir="${base_dir}/${dataset_name}/${current_run_name}"
    # Create the directory if it doesn't exist
    mkdir -p "${model_train_base_dir}"

    ########## Training command ##########
    local CMD="python train.py --experiment_id $experiment_id --num_epochs $num_epochs --train_batch_size $train_batch_size --model_name $model ${dataset_arg} --base_dir ${model_train_base_dir}"

    # Append flags if enabled
    CMD+=" $(add_flag --use_fp16 $USE_FP16)"
    CMD+=" $(add_flag --add_special_tokens_and_normalize $ADD_SPECIAL_TOKENS_AND_NORMALIZE)"
    CMD+=" $(add_flag --preprocess $PREPROCESS)"
    CMD+=" $(add_flag --train_tokenizer $TRAIN_TOKENIZER)"
    CMD+=" $(add_flag --augmentate $AUGMENTATE)"
    CMD+=" $(add_flag --augmentate_valid $AUGMENTATE_VALID)"
    CMD+=" $(add_flag --disable_mlflow $DISABLE_MLFLOW)"
    CMD+=" $(add_flag --concat_train_test $CONCAT_TRAIN_TEST)"
    CMD+=" $(add_flag --use_class_weight $USE_CLASS_WEIGHT)"
    CMD+=" $(add_flag --no_calc_per_class_f1 $NO_CALC_PER_CLASS_F1)"
    CMD+=" $(add_flag --do_not_evaluate $DO_NOT_EVALUATE)"
    CMD+=" $(add_flag --val_2_times $VAL_2_TIMES)"
    CMD+=" $(add_flag --generate_synthetic_train $GENERATE_SYNTHETIC_TRAIN)"
    CMD+=" $(add_flag --generate_synthetic_valid $GENERATE_SYNTHETIC_VALID)"
    CMD+=" $(add_flag --make_custom_tokenizer $MAKE_CUSTOM_TOKENIZER)"
    CMD+=" $(add_flag --drain $DRAIN)"
    CMD+=" $(add_flag --use_mlm_model $USE_MLM_MODEL)"

    # Append other parameters
    CMD+=" --max_length $max_length"
    CMD+=" --weight_decay $weight_decay"
    CMD+=" --logging_steps $logging_steps"
    CMD+=" --eval_accumulation_steps $eval_accumulation_steps"
    CMD+=" --metric_for_best_model $metric_for_best_model"
    CMD+=" --dataloader_num_workers $dataloader_num_workers"
    CMD+=" --vocab_size $vocab_size"
    CMD+=" --run_name $current_run_name"

    if [ "$DOWNLOAD_FOR_RETRAIN" -eq 1 ]; then
        CMD+=" --download_for_retrain $run_id"
        CMD+=" --retrain_epochs $retrain_epochs"
    fi

    if [ "$PATH_FILTER_CATEGORIES" -eq 1 ]; then
        CMD+=" --path_filter_classes $categories_path"
    fi

    echo "#################################################"
    echo "Running command: $CMD"
    eval "$CMD" 2>&1 | sed -r 's/\x1B\[[0-9;?]*[A-Za-z]//g' \
        > "${model_train_base_dir}/console_output.txt"
    echo "#################################################"

    ########################################
    # Log trained dataset
    # echo "$dataset_name" >> "$log_file"
}

###############################
# Training loop (usually no need to change)
###############################

# Loop over the model names to train
for model in "${models[@]}"
do
    # Discover dataset entries (could be: single json file, HF dataset dir, or folder with sub-datasets)
    dataset_paths=()
    if [ -f "$dataset_root" ]; then
        dataset_paths+=("$dataset_root")
    elif [ -d "$dataset_root" ]; then
        # If dataset_root looks like a HF dataset saved to disk (arrow files or dataset_info), treat it as single dataset
        if compgen -G "$dataset_root"/*.arrow > /dev/null || [ -f "$dataset_root/dataset_info.json" ] || [ -f "$dataset_root/dataset_dict.json" ] || [ -f "$dataset_root/dataset_infos.json" ]; then
            dataset_paths+=("$dataset_root")
        else
            # Otherwise treat each immediate subdirectory as a dataset candidate
            for p in "$dataset_root"/*; do
                if [ -d "$p" ]; then
                    dataset_paths+=("$p")
                fi
            done
        fi
    else
        echo "Dataset root not found: $dataset_root"
        continue
    fi

    # Loop over discovered dataset paths
    for dataset_folder in "${dataset_paths[@]}"
    do
        # Normalize path and set defaults
        dataset_folder="${dataset_folder%/}"
        dataset_arg=""
        dataset_name="$(basename "$dataset_folder")"

        echo "#################################################"
        echo "Checking dataset: $dataset_folder"

        # If it's a file (likely a single JSON)
        if [ -f "$dataset_folder" ]; then
            if [[ "$dataset_folder" == *.json ]]; then
                dataset_arg="--json_dataset $dataset_folder"
                dataset_name="$(basename "$dataset_folder" .json)"
            else
                echo "Unsupported dataset file type: $dataset_folder -- skipping"
                continue
            fi

        # If it's a directory, decide whether it's a HF dataset dir, a folder with json files, or a folder of subdatasets
        elif [ -d "$dataset_folder" ]; then
            # HF dataset saved to disk?
            if compgen -G "$dataset_folder"/*.arrow > /dev/null || [ -f "$dataset_folder/dataset_info.json" ] || [ -f "$dataset_folder/dataset_dict.json" ] || [ -f "$dataset_folder/dataset_infos.json" ]; then
                dataset_arg="--dataset_path $dataset_folder"

            else
                # Count json files in the top-level of this directory
                mapfile -t json_files < <(find "$dataset_folder" -maxdepth 1 -type f -name '*.json' -print)
                json_count=${#json_files[@]}

                if [ "$json_count" -eq 0 ]; then
                    # No top-level json files - treat as folder-of-datasets (each subdir may contain jsons)
                    dataset_arg="--dataset_folder $dataset_folder"
                elif [ "$json_count" -eq 1 ]; then
                    # Single json file inside this directory - use it as single json dataset
                    dataset_arg="--json_dataset ${json_files[0]}"
                else
                    # Multiple json files - pass the folder to --dataset_folder
                    dataset_arg="--dataset_folder $dataset_folder"
                fi
            fi
        else
            echo "Skipping unknown dataset path: $dataset_folder"
            continue
        fi

        # Read USE flags from .env and adjust arrays accordingly
        USE_AUG_WHITELIST=$(get_env_value "USE_AUGMENTATION_WHITELIST")
        USE_WHOLE_WHITELIST=$(get_env_value "USE_WHOLE_AUGMENTATION_WHITELIST")
        if [ "$USE_AUG_WHITELIST" != "True" ]; then
            WHITELIST_VALUES=()
        fi
        if [ "$USE_WHOLE_WHITELIST" != "True" ]; then
            WHOLE_LOG_WHITELIST_VALUES=()
        fi

        # Determine which augmentations to run
        if [ ${#WHITELIST_VALUES[@]} -gt 0 ] && [ ${#WHOLE_LOG_WHITELIST_VALUES[@]} -gt 0 ]; then
            # Both entity and whole
            for whitelist in "${WHITELIST_VALUES[@]}"; do
                for aug_part in "${PART_VALUES[@]}"; do
                    for whole_whitelist in "${WHOLE_LOG_WHITELIST_VALUES[@]}"; do
                        run_training "$whitelist" "$aug_part" "$whole_whitelist"
                    done
                done
            done
        elif [ ${#WHITELIST_VALUES[@]} -gt 0 ]; then
            # Only entity
            for whitelist in "${WHITELIST_VALUES[@]}"; do
                aug_part="${PART_VALUES[0]}"
                run_training "$whitelist" "$aug_part" ""
            done
        elif [ ${#WHOLE_LOG_WHITELIST_VALUES[@]} -gt 0 ]; then
            # Only whole
            for whole_whitelist in "${WHOLE_LOG_WHITELIST_VALUES[@]}"; do
                aug_part="${PART_VALUES[0]}"
                run_training "" "$aug_part" "$whole_whitelist"
            done
        else
            # No augmentation
            aug_part="${PART_VALUES[0]}"
            run_training "" "$aug_part" ""
        fi
    done
done

###############################
# User notes:
# - This script will run WHITELIST_VALUES from 1 to 14 and WHOLE_LOG_WHITELIST_VALUES from 1 to 11 sequentially, allowing combinations of entity-level and whole-log augmentations, or separately if one array is empty.
# - To run only whole augmentation, set WHITELIST_VALUES=() (empty array).
# - To run only entity augmentation, set WHOLE_LOG_WHITELIST_VALUES=() (empty array).
# - To run combinations, set both arrays.
# - To run: make executable and run this script (see instructions).
###############################
