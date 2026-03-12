#!/usr/bin/env python3
"""
Minimal training script for distilled models.
Uses scikit-learn — the only Python dependency needed.

Reads JSONL train/val files, trains a model, saves it.
"""
import argparse
import json
import os
import sys
import pickle

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def train_classifier(train_data, val_data, labels, output_dir):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline

    X_train = [r['text'] for r in train_data]
    y_train = [r['label'] for r in train_data]
    X_val = [r['text'] for r in val_data]
    y_val = [r['label'] for r in val_data]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Labels: {labels}")
    sys.stdout.flush()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])

    print("Training...")
    sys.stdout.flush()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, zero_division=0)

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"\n{report}")
    sys.stdout.flush()

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    meta = {
        'task_type': 'classification',
        'labels': labels,
        'accuracy': acc,
        'train_size': len(X_train),
        'val_size': len(X_val)
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {model_path}")
    return acc

def train_extraction(train_data, val_data, output_dir):
    # For extraction tasks, train per-field classifiers or a simple NER-like approach
    # Placeholder — can be expanded
    print("Extraction training not yet implemented — saving data for manual pipeline")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)

def main():
    parser = argparse.ArgumentParser(description='Train a distilled model')
    parser.add_argument('--train', required=True, help='Path to training JSONL')
    parser.add_argument('--val', required=True, help='Path to validation JSONL')
    parser.add_argument('--output', required=True, help='Output directory for model')
    parser.add_argument('--task-type', required=True, choices=['classification', 'extraction', 'regression', 'sequence-labeling'])
    parser.add_argument('--labels', help='Comma-separated labels')
    args, extra = parser.parse_known_args()

    train_data = read_jsonl(args.train)
    val_data = read_jsonl(args.val)

    if args.task_type == 'classification':
        labels = args.labels.split(',') if args.labels else list(set(r['label'] for r in train_data))
        train_classifier(train_data, val_data, labels, args.output)
    elif args.task_type == 'extraction':
        train_extraction(train_data, val_data, args.output)
    else:
        print(f"Task type '{args.task_type}' not yet supported")
        sys.exit(1)

if __name__ == '__main__':
    main()
