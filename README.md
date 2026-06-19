# deepAugmentator

Neural contextual generation (NCG) component of the security log augmentation framework. Fine-tunes transformer-based language models on security log datasets and uses them to fill masked entity spans in log records.

## Structure

```
deepAugmentator/
├── pipeline/
│   ├── pipeline.py               # Main DeepAugmentator class — loads model from MLflow, runs augmentation
│   ├── AI_augmentator.py         # Fill-mask augmentation (MLM models: RoBERTa, ALBERT, MobileBERT, ELECTRA)
│   └── AI_Ollama_augmentator.py  # Next-word prediction via Ollama API (LLaMA, DeepSeek)
├── train_scripts/
│   ├── MLM_training.py           # Fine-tuning for Masked Language Models
│   ├── gpt_training.py           # Fine-tuning for GPT-2
│   ├── llama_training.py         # Fine-tuning for LLaMA 3.2
│   ├── smollm2_instruct_training.py  # Fine-tuning for SmolLM2
│   └── run.sh                    # Training launcher — select model and dataset path here
└── MLflow_model_load/
    └── MLM_models.ipynb          # Notebook for loading and testing models from MLflow registry
```

## How it works

Log records with `<mask>` tokens in entity spans are passed to the model. The model predicts a replacement value from the surrounding log context. Entity offsets are recalculated after substitution.

Two generation modes are supported:
- **MLM (fill-mask)** — bidirectional context via RoBERTa, ALBERT, MobileBERT, ELECTRA
- **NWP (next-word prediction)** — autoregressive generation via Ollama API (LLaMA 3.2, DeepSeek)

## Training

Edit `run.sh` to select a model and dataset path, then run:

```bash
cd train_scripts
bash run.sh
```

Trained models are saved to `MLM_trained_models/<model>/<version>/` and tracked in MLflow.

## Inference

```python
from pipeline.pipeline import DeepAugmentator

augmentator = DeepAugmentator(
    model_path="path/to/model",
    tokenizer_path="path/to/tokenizer",
    keep_mask=True
)
```

Or load directly from MLflow model registry — see `MLflow_model_load/MLM_models.ipynb`.

## Dependencies

- `transformers`, `torch`, `datasets`, `mlflow`, `ollama`
