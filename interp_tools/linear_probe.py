import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import csv
import os
from . import __init__
from typing import Optional


def _ensure_results_dirs(base: str = 'data/linear_probe_results') -> Dict[str, str]:
    acts = os.path.join(base, 'activations')
    regs = os.path.join(base, 'regression')
    os.makedirs(acts, exist_ok=True)
    os.makedirs(regs, exist_ok=True)
    os.makedirs(base, exist_ok=True)
    return {'base': base, 'activations': acts, 'regression': regs}


def _accuracy_csv_path() -> str:
    base = 'data/linear_probe_results'
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, 'accuracy.csv')


def _load_accuracy_rows() -> List[Dict[str, str]]:
    path = _accuracy_csv_path()
    if not os.path.exists(path):
        return []
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def _save_accuracy_rows(rows: List[Dict[str, str]]) -> None:
    path = _accuracy_csv_path()
    header = ['index', 'layer', 'accuracy', 'precision', 'recall']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in header})


def _has_accuracy_entry(existing_rows: List[Dict[str, str]], index: str, layer: int) -> bool:
    for r in existing_rows:
        if r.get('index', '') == str(index) and str(r.get('layer', '')) == str(layer):
            return True
    return False


def extract_layer_activations(text: str, model, tokenizer, layer_num: int):
    """Extract the last non-padding token activation from `layer_num` for `text`.

    Returns a 1D torch tensor (hidden_size,) or None on failure.
    """
    activations_list = []

    def save_activations(module, input, output):
        activations_list.append(output[0].detach())

    target_layer = model.model.layers[layer_num]
    hook_handle = target_layer.register_forward_hook(save_activations)

    messages_for_activation = [{'role': 'user', 'content': text}]
    tokenized_output = tokenizer.apply_chat_template(
        messages_for_activation,
        add_generation_prompt=False,
        return_tensors='pt',
    )

    if isinstance(tokenized_output, torch.Tensor):
        input_ids = tokenized_output.to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
    elif hasattr(tokenized_output, 'keys') and 'input_ids' in tokenized_output:
        input_ids = tokenized_output['input_ids'].to(model.device)
        attention_mask = tokenized_output.get('attention_mask', torch.ones_like(input_ids)).to(model.device)
    else:
        hook_handle.remove()
        return None

    model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with torch.no_grad():
        model(**model_inputs)

    hook_handle.remove()

    if activations_list:
        full_sequence_activations = activations_list[0]
        if full_sequence_activations.ndim == 2:
            full_sequence_activations = full_sequence_activations.unsqueeze(0)
        elif full_sequence_activations.ndim == 1:
            full_sequence_activations = full_sequence_activations.unsqueeze(0).unsqueeze(0)
        actual_sequence_length = attention_mask[0].sum().item()
        final_activation = full_sequence_activations[0, actual_sequence_length - 1, :]
        return final_activation
    return None

def collect_activations(prompts: List[str], labels: List[int], model, tokenizer, probing_layers: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    all_activations_by_layer = {}
    all_labels_by_layer = {}
    for layer_num in probing_layers:
        layer_acts = []
        layer_labels = []
        for prompt, label in zip(prompts, labels):
            act = extract_layer_activations(prompt, model, tokenizer, layer_num)
            if act is None:
                continue
            if hasattr(act, 'cpu'):
                arr = act.float().cpu().numpy()
            else:
                arr = np.array(act)
            layer_acts.append(arr)
            layer_labels.append(label)
        all_activations_by_layer[layer_num] = np.array(layer_acts)
        all_labels_by_layer[layer_num] = np.array(layer_labels)
    return all_activations_by_layer, all_labels_by_layer

def run_linear_probes(all_activations_by_layer: Dict[int, np.ndarray], all_labels_by_layer: Dict[int, np.ndarray], probing_layers: List[int], test_size: float = 0.2):
    results = {}
    for layer_num in probing_layers:
        X = all_activations_by_layer.get(layer_num)
        y = all_labels_by_layer.get(layer_num)
        if X is None or len(X) < 4:
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        clf = LogisticRegression(solver='liblinear', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        results[layer_num] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'train': X_train.shape[0], 'test': X_test.shape[0]}
    return results

__all__ = ['extract_layer_activations', 'collect_activations', 'run_linear_probes']


def run_probes_from_hallucination_rows(
    rows: List[Dict[str, str]],
    model,
    tokenizer,
    probing_layers: List[int],
    threshold: float = 0.5,
    save_activations: bool = False,
    save_regression: bool = False,
    recompute: bool = False,
    random_state: int = 42,
) -> None:
    """Run linear-probe logistic regressions using data from hallucination_frequencies rows.

    Args:
        rows: list of dicts loaded from data/hallucination_frequencies/<language>.csv
        model, tokenizer: model/tokenizer used to extract activations
        probing_layers: list of layer numbers to probe
        threshold: hallucination_rate >= threshold -> positive label
        save_activations: if True save per-index activations tensor to data/linear_probe_results/activations/activations_<index>.pt
        save_regression: if True save regression weights per-index to data/linear_probe_results/regression/regression_<index>.pt
        recompute: if False, skip entries already present in data/linear_probe_results/accuracy.csv for the same index+layer
    """
    dirs = _ensure_results_dirs()

    # Prepare prompts and labels
    prompts = []
    labels = []
    indices = []
    rates = []
    for r in rows:
        idx = r.get('index') or r.get('idx') or ''
        try:
            rate = float(r.get('hallucination_rate', r.get('rate', 0.0)) or 0.0)
        except Exception:
            rate = 0.0
        prompt = r.get('prompt', '')
        label = 1 if rate >= threshold else 0
        indices.append(str(idx))
        prompts.append(prompt)
        labels.append(label)
        rates.append(rate)

    if len(prompts) == 0:
        return

    # Collect activations for all prompts for requested layers
    all_acts_by_layer, all_labels_by_layer = collect_activations(prompts, labels, model, tokenizer, probing_layers)

    existing = _load_accuracy_rows()
    new_rows = list(existing)

    # For each sample index, do leave-one-out training and prediction per layer
    num_samples = len(prompts)
    for i, idx in enumerate(indices):
        # build per-index activations matrix if requested
        if save_activations:
            # build tensor with shape (num_layers, hidden_size)
            layer_tensors = []
            for layer in probing_layers:
                acts = all_acts_by_layer.get(layer)
                if acts is None or i >= acts.shape[0]:
                    continue
                vec = acts[i]
                layer_tensors.append(torch.tensor(vec))
            if layer_tensors:
                stacked = torch.stack(layer_tensors)
                act_path = os.path.join(dirs['activations'], f'activations_{idx}.pt')
                torch.save(stacked, act_path)

        # for each layer, train on others and predict this one
        for layer in probing_layers:
            X = all_acts_by_layer.get(layer)
            y = np.array(labels)
            if X is None or X.shape[0] < 3:
                continue

            if not recompute and _has_accuracy_entry(existing, idx, layer):
                continue

            # Leave-one-out: train on all except i
            mask = np.ones(X.shape[0], dtype=bool)
            mask[i] = False
            X_train = X[mask]
            y_train = y[mask]
            X_test = X[~mask].reshape(1, -1)
            y_test = np.array([y[i]])

            # Need at least two classes in training set
            if len(np.unique(y_train)) < 2:
                # cannot train; record empty metrics
                acc = 0.0
                prec = 0.0
                rec = 0.0
                clf = None
            else:
                clf = LogisticRegression(solver='liblinear', random_state=random_state)
                try:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = float(accuracy_score(y_test, y_pred))
                    prec = float(precision_score(y_test, y_pred, zero_division=0))
                    rec = float(recall_score(y_test, y_pred, zero_division=0))
                except Exception:
                    acc = 0.0
                    prec = 0.0
                    rec = 0.0
                    clf = None

            # Save regression weights per-index if requested
            if save_regression and clf is not None:
                reg_path = os.path.join(dirs['regression'], f'regression_{idx}.pt')
                to_save = {'layer': layer, 'coef': torch.tensor(clf.coef_), 'intercept': torch.tensor(clf.intercept_)}
                torch.save(to_save, reg_path)

            new_rows.append({
                'index': str(idx),
                'layer': str(layer),
                'accuracy': str(acc),
                'precision': str(prec),
                'recall': str(rec)
            })

    # write accuracy CSV (merge existing + new, preferring new rows for duplicates)
    # Build map to avoid duplicates
    merged = {}
    for r in new_rows:
        key = (r.get('index', ''), r.get('layer', ''))
        merged[key] = r

    final_rows = [v for v in merged.values()]
    _save_accuracy_rows(final_rows)
