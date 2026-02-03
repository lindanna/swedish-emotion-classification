"""
Fine-tuning script for KB-BERT on BRIGHTER emotion dataset.

Handles class imbalance through:
1. Upsampling of rare labels (fear, surprise)
2. Weighted loss (optional, via pos_weight in BCEWithLogitsLoss)

Usage:
    python train.py
    python train.py --epochs 3 --batch_size 32
    python train.py --use_pos_weight
    python train.py --help
"""

from transformers.utils import logging
logging.set_verbosity_error()

import argparse
import numpy as np
from collections import Counter

from sklearn.metrics import f1_score, accuracy_score
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets

# Label names
LABEL_NAMES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
NUM_LABELS = len(LABEL_NAMES)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune BERT for emotion classification')
    
    # Model & data
    parser.add_argument('--model_name', type=str, default='KBLab/bert-base-swedish-cased',
                        help='Pretrained model name or path')
    parser.add_argument('--dataset', type=str, default='brighter-dataset/BRIGHTER-emotion-categories',
                        help='Dataset name or path')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    
    # Class imbalance handling
    parser.add_argument('--use_pos_weight', action='store_true',
                        help='Use weighted loss to handle class imbalance')
    parser.add_argument('--upsample_fear', type=int, default=4,
                        help='Upsampling factor for fear label (default: 4)')
    parser.add_argument('--upsample_surprise', type=int, default=2,
                        help='Upsampling factor for surprise label (default: 2)')
    
    # Paths
    parser.add_argument('--output_dir', type=str, default='training/bertbright',
                        help='Directory for training checkpoints')
    parser.add_argument('--save_dir', type=str, default='models/bertbright',
                        help='Directory to save final model')
    
    return parser.parse_args()

def print_datainfo(dataset, label_names=LABEL_NAMES):

    # --- dataset size ---
    print(f"Dataset size: {len(dataset)} examples")

    # --- label distribution ---
    all_labels = [label for sample in dataset["emotions"] for label in sample]
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())

    print("\nLabel distribution:")
    if label_names is None:
        label_names = sorted(label_counts.keys())

    for lbl in label_names:
        count = label_counts.get(lbl, 0)
        pct = (count / total) * 100 if total > 0 else 0.0
        print(f"  {lbl:10s}: {count:5d} ({pct:5.1f}%)")

def upsample_label(ds, label_name, factor=2):
    idx = [i for i, labs in enumerate(ds['emotions']) if label_name in labs]
    if not idx or factor <= 1:
        return ds
    copies = [ds.select(idx) for _ in range(factor-1)]
    return concatenate_datasets([ds] + copies).shuffle(seed=42)

def split_and_upsample(ds_path,upsample_dict):

    ds = load_dataset(ds_path, "swe")

    # split dataset into new parts 
    full_ds = concatenate_datasets([ds["train"], ds["dev"], ds["test"]])
    full_ds = full_ds.shuffle(seed=42)

    train_frac = 0.8
    test_frac = 0.1
    val_frac = 0.1

    train_test_split = full_ds.train_test_split(test_size=(1 - train_frac), seed=42)
    ds_train = train_test_split["train"]
    ds_temp = train_test_split["test"]

    val_test_split = ds_temp.train_test_split(test_size=test_frac / (test_frac + val_frac), seed=42)

    ds_val = val_test_split["train"]
    ds_test = val_test_split["test"]

    # upsample the labels with low occurence
    for lbl,f in upsample_dict.items():
        ds_train = upsample_label(ds_train,lbl,factor=f)

    print_datainfo(full_ds)

    return ds_train,ds_val,ds_test

def preprocess(examples,tokenizer):
    # tokenize texts
    tokenized = tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    ) 
    
    # convert labels to multi-hot encoding
    label_names = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    labels = []
    for label_list in examples['emotions']:
        multi_hot = [1.0 if label in label_list else 0.0 for label in label_names]
        labels.append(multi_hot)
    
    tokenized['labels'] = labels
    return tokenized

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    # apply sigmoid and threshold at 0.5
    predictions = (torch.sigmoid(torch.tensor(predictions)) > 0.5).numpy()
    
    # calculate metrics
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_per_label = f1_score(labels, predictions, average=None, zero_division=0)

    accuracy = accuracy_score(labels, predictions)

    # print per-label F1s
    print("\nPer-label F1 scores:")
    for name, f1 in zip(LABEL_NAMES, f1_per_label):
        print(f"  {name:10s}: {f1:.4f}")
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': accuracy
    }

def compute_pos_weight(ds_train, label_names=LABEL_NAMES):
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.
    Args:
        ds_train: Training dataset
        label_names: List of label names   
    Returns:
        torch.Tensor: Weight for each label (neg_count / pos_count)
    """
    # build counts
    counts = np.zeros(len(label_names), dtype=np.int64)
    for labs in ds_train['emotions']:
        for i, name in enumerate(label_names):
            if name in labs:
                counts[i] += 1
    N = len(ds_train)
    pos = counts.astype(np.float32)
    neg = (N - pos).astype(np.float32)
    # avoid divide-by-zero, if a label never appears, set weight 1.0
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1e-6), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)

class BCEWithLogitsTrainer(Trainer):
    """
    Custom Trainer that uses BCEWithLogitsLoss with pos_weight 
    to handle class imbalance in multi-label classification.
    """
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight  # torch.Tensor or None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.pop("labels").float()  # [B, C]
        outputs = model(**inputs)
        logits = outputs.logits  # [B, C]

        # Ensure pos_weight dtype matches logits dtype and lives on the same device
        if self._pos_weight is not None:
            pw = self._pos_weight.to(device=logits.device, dtype=logits.dtype)
        else:
            pw = None

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="mean")
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

def main(args):

    print("Loading tokenizer and model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=6,
        problem_type="multi_label_classification"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)

    print("Loading & preprocessing dataset...")

    # new dataset splits, upsample low frequency labels
    
    # create upsampling dictionary from args
    upsample_dict = {
        'fear': args.upsample_fear,
        'surprise': args.upsample_surprise
    }

    #upsample_dict ={'fear':4,'surprise':2}
    ds_train,ds_val,ds_test = split_and_upsample(args.dataset,upsample_dict)

    # tokenize
    tokenized_train = ds_train.map(
        preprocess, 
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}
    )
    
    tokenized_val = ds_val.map(
        preprocess, 
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}
    )
    
    tokenized_test = ds_test.map(
        preprocess, 
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}
    )    

    #------Training

    if args.use_pos_weight: 
        pos_weight = compute_pos_weight(ds_train)
    else:
        pos_weight = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    print("Initializing trainer...")

    trainer = BCEWithLogitsTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        pos_weight=pos_weight,
        )

    print("Starting training...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_test)
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\nSaving model to {args.save_dir}")
    trainer.save_model(args.save_dir)


if __name__ == '__main__':

    args = parse_args()
    main(args)
