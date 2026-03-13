#!/usr/bin/env python3
"""
Transformer fine-tuning script for feature distillation.
Runtime dependency: transformers, torch (optional: onnx, optimum).

Reads JSONL train/val files, fine-tunes a pretrained model, saves it.
Follows the same CLI contract as train.py so it slots into the existing pipeline.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime


def check_dependencies():
    """Check if required packages are installed. Returns dict of availability."""
    deps = {}
    for name in ['torch', 'transformers']:
        try:
            __import__(name)
            deps[name] = True
        except ImportError:
            deps[name] = False
    # Optional deps
    for name in ['optimum', 'onnx', 'onnxruntime']:
        try:
            __import__(name)
            deps[name] = True
        except ImportError:
            deps[name] = False
    return deps


def detect_device():
    """Detect best available device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        return 'cuda', f'{name} ({mem:.1f}GB)'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps', 'Apple Silicon GPU'
    return 'cpu', 'CPU'


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Supported model presets
MODEL_PRESETS = {
    'distilbert': {
        'name': 'distilbert-base-uncased',
        'max_length': 128,
        'default_lr': 5e-5,
        'default_epochs': 3,
        'params': '66M'
    },
    'tinybert': {
        'name': 'huawei-noah/TinyBERT_General_4L_312D',
        'max_length': 128,
        'default_lr': 5e-5,
        'default_epochs': 5,
        'params': '14M'
    },
    'bert-base': {
        'name': 'bert-base-uncased',
        'max_length': 256,
        'default_lr': 2e-5,
        'default_epochs': 3,
        'params': '110M'
    },
    'roberta': {
        'name': 'roberta-base',
        'max_length': 256,
        'default_lr': 2e-5,
        'default_epochs': 3,
        'params': '125M'
    },
    'minilm': {
        'name': 'microsoft/MiniLM-L12-H384-uncased',
        'max_length': 128,
        'default_lr': 5e-5,
        'default_epochs': 4,
        'params': '33M'
    }
}


def build_dataset(data, tokenizer, label2id, max_length):
    """Build a torch Dataset from JSONL data."""
    import torch

    texts = [r['text'] for r in data]
    labels = [label2id[r['label']] for r in data]

    encodings = tokenizer(texts, truncation=True, padding='max_length',
                          max_length=max_length, return_tensors='pt')

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    return TextDataset(encodings, labels)


def train_transformer(train_data, val_data, labels, output_dir, model_key='distilbert',
                      custom_model=None, epochs=None, lr=None, batch_size=16,
                      max_length=None, warmup_ratio=0.1, weight_decay=0.01,
                      export_onnx=False):
    """Fine-tune a transformer for text classification."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    import numpy as np

    # Resolve model config
    if custom_model:
        model_name = custom_model
        preset = MODEL_PRESETS.get('distilbert')  # defaults
    elif model_key in MODEL_PRESETS:
        preset = MODEL_PRESETS[model_key]
        model_name = preset['name']
    else:
        print(f"Unknown model key: {model_key}. Available: {', '.join(MODEL_PRESETS.keys())}")
        sys.exit(1)

    actual_epochs = epochs or preset.get('default_epochs', 3)
    actual_lr = lr or preset.get('default_lr', 5e-5)
    actual_max_length = max_length or preset.get('max_length', 128)

    device, device_info = detect_device()

    print(f"Model: {model_name}")
    print(f"Device: {device} ({device_info})")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Labels: {labels}")
    print(f"Epochs: {actual_epochs}, LR: {actual_lr}, Batch: {batch_size}, Max length: {actual_max_length}")
    sys.stdout.flush()

    # Build label mapping
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    # Build datasets
    train_dataset = build_dataset(train_data, tokenizer, label2id, actual_max_length)
    val_dataset = build_dataset(val_data, tokenizer, label2id, actual_max_length)

    # Training arguments
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'checkpoints'),
        num_train_epochs=actual_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=actual_lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2,
        logging_steps=max(1, len(train_data) // batch_size // 5),
        report_to='none',
        fp16=(device == 'cuda'),
        use_mps_device=(device == 'mps'),
        dataloader_num_workers=0,
        disable_tqdm=True
    )

    # Metrics
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == label_ids).mean()
        return {'accuracy': float(acc)}

    # Logging callback for TUI progress
    class ProgressCallback:
        def __init__(self):
            self.current_epoch = 0

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.current_epoch = state.epoch or 0
            print(f"__EPOCH_START__:{json.dumps({'epoch': int(self.current_epoch) + 1, 'total': actual_epochs})}")
            sys.stdout.flush()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                print(f"__TRAIN_LOG__:{json.dumps({'step': state.global_step, 'loss': logs.get('loss', 0), 'lr': logs.get('learning_rate', 0)})}")
                sys.stdout.flush()

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                acc = metrics.get('eval_accuracy', 0)
                loss = metrics.get('eval_loss', 0)
                print(f"__EVAL_LOG__:{json.dumps({'epoch': int(state.epoch or 0), 'accuracy': acc, 'loss': loss})}")
                sys.stdout.flush()

    # HuggingFace Trainer uses callbacks as objects with on_* methods
    # We need to wrap our simple class to match the TrainerCallback interface
    from transformers import TrainerCallback

    class TUICallback(TrainerCallback):
        def __init__(self):
            self.current_epoch = 0

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.current_epoch = int(state.epoch or 0)
            print(f"__EPOCH_START__:{json.dumps({'epoch': self.current_epoch + 1, 'total': actual_epochs})}")
            sys.stdout.flush()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                print(f"__TRAIN_LOG__:{json.dumps({'step': state.global_step, 'loss': round(logs.get('loss', 0), 4), 'lr': logs.get('learning_rate', 0)})}")
                sys.stdout.flush()

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                acc = metrics.get('eval_accuracy', 0)
                loss = metrics.get('eval_loss', 0)
                print(f"__EVAL_LOG__:{json.dumps({'epoch': int(state.epoch or 0), 'accuracy': round(acc, 4), 'loss': round(loss, 4)})}")
                sys.stdout.flush()

    # Train
    print("Training...")
    sys.stdout.flush()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[TUICallback()]
    )

    start = time.time()
    trainer.train()
    duration = time.time() - start

    # Final evaluation
    results = trainer.evaluate()
    acc = results.get('eval_accuracy', 0)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Training time: {duration:.1f}s")
    sys.stdout.flush()

    # Save model + tokenizer
    model_save_dir = os.path.join(output_dir, 'transformer')
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    # Save meta.json (same format as train.py for compatibility)
    meta = {
        'task_type': 'classification',
        'algorithm': f'transformer:{model_key}',
        'model_name': model_name,
        'feature_mode': 'transformer',
        'labels': labels,
        'accuracy': round(acc, 4),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'epochs': actual_epochs,
        'learning_rate': actual_lr,
        'batch_size': batch_size,
        'max_length': actual_max_length,
        'warmup_ratio': warmup_ratio,
        'weight_decay': weight_decay,
        'device': device,
        'duration_s': round(duration, 1),
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {model_save_dir}")

    # ONNX export
    if export_onnx:
        try:
            onnx_dir = os.path.join(output_dir, 'onnx')
            os.makedirs(onnx_dir, exist_ok=True)

            # Try optimum first (better ONNX export for HF models)
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                ort_model = ORTModelForSequenceClassification.from_pretrained(model_save_dir, export=True)
                ort_model.save_pretrained(onnx_dir)
                tokenizer.save_pretrained(onnx_dir)
                meta['onnx_path'] = onnx_dir
                print(f"ONNX model saved to {onnx_dir} (via optimum)")
            except ImportError:
                # Fallback to torch.onnx
                import torch
                dummy = tokenizer("test", return_tensors='pt', max_length=actual_max_length,
                                  padding='max_length', truncation=True)
                if device == 'cuda':
                    dummy = {k: v.cuda() for k, v in dummy.items()}
                    model.cuda()

                onnx_path = os.path.join(onnx_dir, 'model.onnx')
                torch.onnx.export(
                    model,
                    tuple(dummy.values()),
                    onnx_path,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch', 1: 'seq'},
                        'attention_mask': {0: 'batch', 1: 'seq'},
                        'logits': {0: 'batch'}
                    },
                    opset_version=14
                )
                tokenizer.save_pretrained(onnx_dir)
                meta['onnx_path'] = onnx_path
                print(f"ONNX model saved to {onnx_path} (via torch.onnx)")

            # Update meta with ONNX info
            with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
        sys.stdout.flush()

    print(f"\n__TRANSFORMER_RESULTS__:{json.dumps({'accuracy': round(acc, 4), 'model': model_name, 'device': device, 'duration_s': round(duration, 1), 'epochs': actual_epochs})}")
    sys.stdout.flush()
    return acc


def predict_transformer(model_dir, texts, max_length=128):
    """Load a fine-tuned transformer and predict."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device, _ = detect_device()
    model.to(device)
    model.eval()

    encodings = tokenizer(texts, truncation=True, padding='max_length',
                          max_length=max_length, return_tensors='pt')
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        confidences = probs.max(dim=-1).values

    results = []
    for i, text in enumerate(texts):
        label = model.config.id2label[pred_ids[i].item()]
        results.append({
            'text': text,
            'label': label,
            'confidence': round(confidences[i].item(), 4)
        })

    return results


def list_models():
    """List available model presets."""
    print(json.dumps(MODEL_PRESETS, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a transformer model')
    parser.add_argument('--train', help='Path to training JSONL')
    parser.add_argument('--val', help='Path to validation JSONL')
    parser.add_argument('--output', help='Output directory for model')
    parser.add_argument('--task-type', default='classification')
    parser.add_argument('--labels', help='Comma-separated labels')
    parser.add_argument('--model', default='distilbert', help='Model preset or custom HF model name')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, help='Max sequence length')
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--onnx', action='store_true', help='Export to ONNX format')

    # Predict mode
    parser.add_argument('--predict', help='Path to transformer model dir for prediction')
    parser.add_argument('--input', help='Text to predict (or - for stdin JSONL)')

    # Utility
    parser.add_argument('--check-deps', action='store_true', help='Check dependency availability')
    parser.add_argument('--list-models', action='store_true', help='List available model presets')
    parser.add_argument('--detect-device', action='store_true', help='Detect compute device')

    args = parser.parse_args()

    # Utility modes
    if args.check_deps:
        deps = check_dependencies()
        print(json.dumps(deps))
        return

    if args.list_models:
        list_models()
        return

    if args.detect_device:
        deps = check_dependencies()
        if not deps.get('torch'):
            print(json.dumps({'device': 'unavailable', 'info': 'torch not installed'}))
            return
        device, info = detect_device()
        print(json.dumps({'device': device, 'info': info}))
        return

    # Predict mode
    if args.predict:
        deps = check_dependencies()
        if not deps.get('torch') or not deps.get('transformers'):
            print("Error: torch and transformers are required for prediction.", file=sys.stderr)
            print("Install with: pip3 install torch transformers", file=sys.stderr)
            sys.exit(1)

        model_dir = args.predict
        ml = args.max_length or 128

        if args.input == '-':
            texts = [json.loads(line)['text'] for line in sys.stdin if line.strip()]
        else:
            texts = [args.input]

        results = predict_transformer(model_dir, texts, ml)
        print(json.dumps(results, indent=2))
        return

    # Training mode
    if not args.train or not args.val or not args.output:
        parser.error('Training requires --train, --val, and --output')

    # Check deps before training
    deps = check_dependencies()
    if not deps.get('torch'):
        print("Error: PyTorch is required for transformer training.", file=sys.stderr)
        print("Install with: pip3 install torch", file=sys.stderr)
        print("  GPU (CUDA): pip3 install torch --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        print("  CPU only:   pip3 install torch --index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
        sys.exit(1)
    if not deps.get('transformers'):
        print("Error: HuggingFace Transformers is required.", file=sys.stderr)
        print("Install with: pip3 install transformers", file=sys.stderr)
        sys.exit(1)

    train_data = read_jsonl(args.train)
    val_data = read_jsonl(args.val)
    labels = args.labels.split(',') if args.labels else list(set(r['label'] for r in train_data))

    # Determine if it's a preset or custom model
    model_key = args.model
    custom_model = None
    if model_key not in MODEL_PRESETS:
        custom_model = model_key
        model_key = 'custom'

    train_transformer(
        train_data, val_data, labels, args.output,
        model_key=model_key if model_key != 'custom' else 'distilbert',
        custom_model=custom_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        export_onnx=args.onnx
    )


if __name__ == '__main__':
    main()
