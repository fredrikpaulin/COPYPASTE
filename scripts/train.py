#!/usr/bin/env python3
"""
Training script for distilled models.
Uses scikit-learn — the only hard Python dependency.
Optional: skl2onnx for ONNX export.

Reads JSONL train/val files, trains a model, saves it.
"""
import argparse
import json
import os
import sys
import pickle
from datetime import datetime

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_pipeline(algorithm='logistic_regression', params=None, use_embeddings=False):
    """Build a sklearn Pipeline with the given algorithm and optional params."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    if algorithm == 'logistic_regression':
        clf = LogisticRegression(
            max_iter=int((params or {}).get('max_iter', 1000)),
            C=float((params or {}).get('C', 1.0))
        )
    elif algorithm == 'svm':
        clf = LinearSVC(
            max_iter=int((params or {}).get('max_iter', 1000)),
            C=float((params or {}).get('C', 1.0))
        )
    elif algorithm == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=int((params or {}).get('n_estimators', 100)),
            max_depth=int((params or {}).get('max_depth', 0)) or None,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if use_embeddings:
        # No vectorizer — features are pre-computed embedding vectors
        return Pipeline([('clf', clf)])

    tfidf_params = {
        'max_features': int((params or {}).get('max_features', 10000)),
        'ngram_range': (1, 2)
    }

    return Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', clf)
    ])


def load_embeddings(path):
    """Load pre-computed embeddings from a JSONL file. Each row: {"embedding": [...]}"""
    import numpy as np
    rows = read_jsonl(path)
    return np.array([r['embedding'] for r in rows])


def reduce_dimensions(X, method='pca', n_components=50):
    """Reduce embedding dimensions via PCA or truncated SVD."""
    if method == 'pca':
        from sklearn.decomposition import PCA
        n = min(n_components, X.shape[0], X.shape[1])
        reducer = PCA(n_components=n)
    else:
        from sklearn.decomposition import TruncatedSVD
        n = min(n_components, X.shape[1] - 1)
        reducer = TruncatedSVD(n_components=n)

    print(f"Reducing dimensions: {X.shape[1]} -> {n} ({method})")
    sys.stdout.flush()
    return reducer.fit_transform(X), reducer

def evaluate_model(pipeline, X_val, y_val):
    """Evaluate and return accuracy + per-label report."""
    from sklearn.metrics import classification_report, accuracy_score
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, zero_division=0)
    return acc, y_pred, report

def train_classifier(train_data, val_data, labels, output_dir, export_onnx=False, algorithm='logistic_regression', params=None,
                     train_embeddings=None, val_embeddings=None, dim_reduce=None, n_components=50):
    y_train = [r['label'] for r in train_data]
    y_val = [r['label'] for r in val_data]

    use_embeddings = train_embeddings is not None
    if use_embeddings:
        import numpy as np
        X_train = train_embeddings
        X_val = val_embeddings
        feature_mode = 'embeddings'

        # Optional dimensionality reduction
        reducer = None
        if dim_reduce:
            X_train, reducer = reduce_dimensions(X_train, method=dim_reduce, n_components=n_components)
            X_val = reducer.transform(X_val)
    else:
        X_train = [r['text'] for r in train_data]
        X_val = [r['text'] for r in val_data]
        feature_mode = 'tfidf'

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Labels: {labels}")
    print(f"Algorithm: {algorithm}")
    print(f"Features: {feature_mode}")
    sys.stdout.flush()

    pipeline = build_pipeline(algorithm, params, use_embeddings=use_embeddings)

    print("Training...")
    sys.stdout.flush()
    pipeline.fit(X_train, y_train)

    acc, y_pred, report = evaluate_model(pipeline, X_val, y_val)

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"\n{report}")
    sys.stdout.flush()

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pkl')

    # For embedding models, save the reducer alongside the pipeline
    model_artifact = pipeline
    if use_embeddings and dim_reduce and reducer is not None:
        model_artifact = {'pipeline': pipeline, 'reducer': reducer, 'feature_mode': 'embeddings'}
    with open(model_path, 'wb') as f:
        pickle.dump(model_artifact, f)

    meta = {
        'task_type': 'classification',
        'algorithm': algorithm,
        'feature_mode': feature_mode,
        'labels': labels,
        'accuracy': acc,
        'train_size': len(y_train),
        'val_size': len(y_val),
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    if params:
        meta['params'] = params
    if dim_reduce:
        meta['dim_reduce'] = dim_reduce
        meta['n_components'] = n_components

    # ONNX export
    if export_onnx:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import StringTensorType
            onnx_model = convert_sklearn(
                pipeline,
                'distilled_classifier',
                initial_types=[('text', StringTensorType([None, 1]))]
            )
            onnx_path = os.path.join(output_dir, 'model.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            meta['onnx_path'] = onnx_path
            print(f"ONNX model saved to {onnx_path}")
        except ImportError:
            print("Warning: skl2onnx not installed, skipping ONNX export")
            print("Install with: pip install skl2onnx")
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
        sys.stdout.flush()

    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {model_path}")
    return acc

def compare_classifiers(train_data, val_data, labels, output_dir, algorithms=None):
    """Train multiple algorithms and compare them. Returns results sorted by accuracy."""
    if algorithms is None:
        algorithms = ['logistic_regression', 'svm', 'random_forest']

    X_train = [r['text'] for r in train_data]
    y_train = [r['label'] for r in train_data]
    X_val = [r['text'] for r in val_data]
    y_val = [r['label'] for r in val_data]

    print(f"Comparing {len(algorithms)} algorithms on {len(X_train)} train / {len(X_val)} val samples")
    sys.stdout.flush()

    results = []
    for algo in algorithms:
        print(f"\n--- {algo} ---")
        sys.stdout.flush()
        try:
            pipeline = build_pipeline(algo)
            pipeline.fit(X_train, y_train)
            acc, _, report = evaluate_model(pipeline, X_val, y_val)
            print(f"Accuracy: {acc:.4f}")
            print(report)
            sys.stdout.flush()
            results.append({'algorithm': algo, 'accuracy': acc, 'pipeline': pipeline})
        except Exception as e:
            print(f"Failed: {e}")
            sys.stdout.flush()
            results.append({'algorithm': algo, 'accuracy': 0.0, 'error': str(e)})

    results.sort(key=lambda r: r['accuracy'], reverse=True)
    best = results[0]

    print(f"\n=== COMPARISON RESULTS ===")
    for r in results:
        marker = ' <- best' if r['algorithm'] == best['algorithm'] else ''
        err = f" (error: {r.get('error', '')})" if 'error' in r else ''
        print(f"  {r['algorithm']}: {r['accuracy']:.4f}{marker}{err}")
    sys.stdout.flush()

    # Save best model
    if best.get('pipeline'):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best['pipeline'], f)

        meta = {
            'task_type': 'classification',
            'algorithm': best['algorithm'],
            'labels': labels,
            'accuracy': best['accuracy'],
            'train_size': len(X_train),
            'val_size': len(X_val),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'comparison': [{'algorithm': r['algorithm'], 'accuracy': r['accuracy']} for r in results]
        }
        with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\nBest model ({best['algorithm']}) saved to {model_path}")

    # Structured output for JS parsing
    print(f"\n__COMPARE_RESULTS__:{json.dumps([{'algorithm': r['algorithm'], 'accuracy': r['accuracy']} for r in results])}")
    sys.stdout.flush()
    return results

def hyperparam_search(train_data, val_data, labels, output_dir, algorithm='logistic_regression', grid=None):
    """Grid search over hyperparameters. Returns best params and accuracy."""
    import itertools

    X_train = [r['text'] for r in train_data]
    y_train = [r['label'] for r in train_data]
    X_val = [r['text'] for r in val_data]
    y_val = [r['label'] for r in val_data]

    if grid is None:
        if algorithm == 'logistic_regression':
            grid = {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [500, 1000]}
        elif algorithm == 'svm':
            grid = {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [500, 1000]}
        elif algorithm == 'random_forest':
            grid = {'n_estimators': [50, 100, 200], 'max_depth': [0, 10, 20]}

    keys = list(grid.keys())
    values = list(grid.values())
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"Hyperparameter search: {algorithm}, {len(combos)} combinations")
    sys.stdout.flush()

    results = []
    for i, params in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] {params}")
        sys.stdout.flush()
        try:
            pipeline = build_pipeline(algorithm, params)
            pipeline.fit(X_train, y_train)
            acc, _, _ = evaluate_model(pipeline, X_val, y_val)
            print(f"  Accuracy: {acc:.4f}")
            sys.stdout.flush()
            results.append({'params': params, 'accuracy': acc, 'pipeline': pipeline})
        except Exception as e:
            print(f"  Failed: {e}")
            sys.stdout.flush()

    results.sort(key=lambda r: r['accuracy'], reverse=True)
    best = results[0]

    print(f"\n=== SEARCH RESULTS ===")
    for r in results[:5]:
        marker = ' <- best' if r['params'] == best['params'] else ''
        print(f"  {r['params']}: {r['accuracy']:.4f}{marker}")
    sys.stdout.flush()

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best['pipeline'], f)

    meta = {
        'task_type': 'classification',
        'algorithm': algorithm,
        'labels': labels,
        'accuracy': best['accuracy'],
        'params': best['params'],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'search_results': [{'params': r['params'], 'accuracy': r['accuracy']} for r in results]
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest model saved to {model_path}")
    print(f"\n__SEARCH_RESULTS__:{json.dumps([{'params': r['params'], 'accuracy': r['accuracy']} for r in results])}")
    sys.stdout.flush()
    return results

def train_extraction(train_data, val_data, fields, output_dir):
    """Train per-field binary classifiers for extraction tasks."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    os.makedirs(output_dir, exist_ok=True)
    field_models = {}
    field_accuracies = {}

    for field in fields:
        print(f"\n--- Training extractor for field: {field} ---")
        sys.stdout.flush()

        X_train = [r['text'] for r in train_data]
        y_train = [1 if r.get('fields', {}).get(field) else 0 for r in train_data]
        X_val = [r['text'] for r in val_data]
        y_val = [1 if r.get('fields', {}).get(field) else 0 for r in val_data]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)

        acc = accuracy_score(y_val, pipeline.predict(X_val))
        print(f"  Field '{field}' detection accuracy: {acc:.4f}")
        sys.stdout.flush()

        field_models[field] = pipeline
        field_accuracies[field] = acc

    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(field_models, f)

    meta = {
        'task_type': 'extraction',
        'fields': fields,
        'field_accuracies': field_accuracies,
        'accuracy': sum(field_accuracies.values()) / len(field_accuracies) if field_accuracies else 0,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nAvg field detection accuracy: {meta['accuracy']:.4f}")
    print(f"Model saved to {model_path}")
    sys.stdout.flush()
    return meta['accuracy']

def predict(model_path, texts):
    """Load a model and predict on given texts. Used by inference module."""
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    # Handle extraction models (dict of per-field pipelines)
    if isinstance(pipeline, dict):
        results = []
        for text in texts:
            fields = {}
            for field, model in pipeline.items():
                pred = model.predict([text])[0]
                fields[field] = bool(pred)
            results.append({'text': text, 'fields': fields})
        return results

    predictions = pipeline.predict(texts)
    try:
        probas = pipeline.predict_proba(texts)
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'label': predictions[i],
                'confidence': float(max(probas[i]))
            })
    except AttributeError:
        # LinearSVC doesn't have predict_proba
        results = [{'text': t, 'label': p, 'confidence': None} for t, p in zip(texts, predictions)]
    return results

def main():
    parser = argparse.ArgumentParser(description='Train a distilled model')
    parser.add_argument('--train', help='Path to training JSONL')
    parser.add_argument('--val', help='Path to validation JSONL')
    parser.add_argument('--output', help='Output directory for model')
    parser.add_argument('--task-type', choices=['classification', 'extraction', 'regression', 'sequence-labeling'])
    parser.add_argument('--labels', help='Comma-separated labels')
    parser.add_argument('--fields', help='Comma-separated field names for extraction')
    parser.add_argument('--algorithm', default='logistic_regression',
                       choices=['logistic_regression', 'svm', 'random_forest'],
                       help='Classification algorithm')
    parser.add_argument('--onnx', action='store_true', help='Export to ONNX format')
    parser.add_argument('--compare', action='store_true', help='Compare multiple algorithms')
    parser.add_argument('--search', action='store_true', help='Hyperparameter grid search')
    parser.add_argument('--grid', help='JSON string of hyperparameter grid')
    # Embedding mode
    parser.add_argument('--train-embeddings', help='Path to training embeddings JSONL')
    parser.add_argument('--val-embeddings', help='Path to validation embeddings JSONL')
    parser.add_argument('--dim-reduce', choices=['pca', 'svd'], help='Dimensionality reduction method')
    parser.add_argument('--n-components', type=int, default=50, help='Target dimensions after reduction')
    # Predict mode
    parser.add_argument('--predict', help='Path to model.pkl for prediction')
    parser.add_argument('--input', help='Text to predict (or - for stdin JSONL)')
    args, extra = parser.parse_known_args()

    # Prediction mode
    if args.predict:
        if args.input == '-':
            texts = [json.loads(line)['text'] for line in sys.stdin if line.strip()]
        else:
            texts = [args.input]
        results = predict(args.predict, texts)
        print(json.dumps(results, indent=2))
        return

    # Training mode
    if not args.train or not args.val or not args.output or not args.task_type:
        parser.error('Training requires --train, --val, --output, and --task-type')

    train_data = read_jsonl(args.train)
    val_data = read_jsonl(args.val)

    if args.task_type == 'classification':
        labels = args.labels.split(',') if args.labels else list(set(r['label'] for r in train_data))

        # Load pre-computed embeddings if provided
        train_emb = None
        val_emb = None
        if args.train_embeddings and args.val_embeddings:
            train_emb = load_embeddings(args.train_embeddings)
            val_emb = load_embeddings(args.val_embeddings)

        if args.compare:
            compare_classifiers(train_data, val_data, labels, args.output)
        elif args.search:
            grid = json.loads(args.grid) if args.grid else None
            hyperparam_search(train_data, val_data, labels, args.output, args.algorithm, grid)
        else:
            params = {}
            i = 0
            while i < len(extra):
                if extra[i].startswith('--'):
                    key = extra[i][2:]
                    if i + 1 < len(extra) and not extra[i+1].startswith('--'):
                        params[key] = extra[i+1]
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            train_classifier(train_data, val_data, labels, args.output,
                           export_onnx=args.onnx, algorithm=args.algorithm, params=params or None,
                           train_embeddings=train_emb, val_embeddings=val_emb,
                           dim_reduce=args.dim_reduce, n_components=args.n_components)
    elif args.task_type == 'extraction':
        fields = args.fields.split(',') if args.fields else []
        if not fields:
            for r in train_data:
                if 'fields' in r and isinstance(r['fields'], dict):
                    fields = list(r['fields'].keys())
                    break
        train_extraction(train_data, val_data, fields, args.output)
    else:
        print(f"Task type '{args.task_type}' not yet supported")
        sys.exit(1)

if __name__ == '__main__':
    main()
