"""DiCo-NLI DeBERTa-v3-large: standard baseline and reversal-aware training.

Run `python train_dico_nli.py --help` for options. No training occurs on import.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

MODEL_NAME = 'microsoft/deberta-v3-large'
LABEL2ID = {'EQUIVALENCE': 0, 'FORWARD_ENTAILMENT': 1, 'BACKWARD_ENTAILMENT': 2, 'NEGATIVE_OTHER': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
REVERSE = [0, 2, 1, 3]


def load_data(path, name):
    if not path.is_file():
        raise FileNotFoundError(f'{name} CSV not found: {path}')
    df = pd.read_csv(path)
    required = {'instance_id', 'pair_id', 'text1', 'text2', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')
    if df[list(required)].isna().any().any():
        raise ValueError(f'{name} contains missing required values')
    invalid = set(df.label.unique()) - set(LABEL2ID)
    if invalid:
        raise ValueError(f'{name} unexpected labels: {invalid}')
    if df.instance_id.duplicated().any():
        raise ValueError(f'{name} has duplicate instance_id values')
    df['labels'] = df.label.map(LABEL2ID).astype(int)
    print(f'{name}: {len(df)} rows, {df.pair_id.nunique()} pairs')
    print(df.label.value_counts().to_string())
    return df


def validate_pairs(df, name):
    issues = []
    for pid, group in df.groupby('pair_id'):
        if len(group) == 1 and int(group.iloc[0].labels) == LABEL2ID['NEGATIVE_OTHER']:
            continue  # Official data contains one direction for these examples.
        if len(group) != 2:
            issues.append(f'{pid}: expected 2 rows, found {len(group)}')
            continue
        a, b = group.iloc[0], group.iloc[1]
        if a.text1 != b.text2 or a.text2 != b.text1 or REVERSE[int(a.labels)] != int(b.labels):
            issues.append(f'{pid}: inconsistent reversed pair')
    print(f'{name}: {len(issues)} pair consistency issues')
    if issues:
        print('Examples:', issues[:5])
    return issues


class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df, self.tokenizer, self.max_length = df.reset_index(drop=True), tokenizer, max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoded = self.tokenizer(row.text1, row.text2, truncation=True, max_length=self.max_length)
        encoded['labels'] = int(row.labels)
        return encoded


class ReversalDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.pairs = []
        self.tokenizer, self.max_length = tokenizer, max_length
        for pid, group in df.groupby('pair_id', sort=False):
            if len(group) > 2:
                raise ValueError(f'{pid}: more than two rows')
            original = group[group.instance_id.astype(str).str.endswith('__original')]
            row = original.iloc[0] if len(original) == 1 else group.iloc[0]
            if len(group) == 2:
                other = group[group.instance_id != row.instance_id].iloc[0]
                if row.text1 != other.text2 or row.text2 != other.text1 or int(other.labels) != REVERSE[int(row.labels)]:
                    raise ValueError(f'{pid}: inconsistent reversed pair')
            self.pairs.append((row.text1, row.text2, int(row.labels)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a, b, label = self.pairs[index]
        tokenize = lambda x, y: self.tokenizer(x, y, truncation=True, max_length=self.max_length)
        return {'original': tokenize(a, b), 'reversed': tokenize(b, a), 'label_original': label, 'label_reversed': REVERSE[label]}


def reversal_collator(tokenizer):
    pad = DataCollatorWithPadding(tokenizer, return_tensors='pt')
    def collate(items):
        return {'original': pad([x['original'] for x in items]),
                'reversed': pad([x['reversed'] for x in items]),
                'labels_original': torch.tensor([x['label_original'] for x in items]),
                'labels_reversed': torch.tensor([x['label_reversed'] for x in items])}
    return collate


def evaluate(model, loader, device):
    model.eval()
    losses, labels, predictions = 0., [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(**batch)
            losses += result.loss.item() * len(batch['labels'])
            labels.extend(batch['labels'].cpu().tolist())
            predictions.extend(result.logits.argmax(-1).cpu().tolist())
    return {'dev_loss': losses / len(labels), 'dev_accuracy': accuracy_score(labels, predictions),
            'dev_weighted_f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'dev_macro_f1': f1_score(labels, predictions, average='macro', zero_division=0)}


def reversal_loss(model, batch, device, weight):
    a = {k: v.to(device) for k, v in batch['original'].items()}
    b = {k: v.to(device) for k, v in batch['reversed'].items()}
    logits_a, logits_b = model(**a).logits, model(**b).logits
    cls = (F.cross_entropy(logits_a, batch['labels_original'].to(device)) +
           F.cross_entropy(logits_b, batch['labels_reversed'].to(device))) / 2
    cons = F.mse_loss(F.softmax(logits_a.float(), -1), F.softmax(logits_b.float(), -1)[:, REVERSE])
    return cls + weight * cons, cls, cons


def train(args, mode, train_df, dev_df, tokenizer, device):
    out = args.output_dir / mode
    out.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')
    dev_loader = DataLoader(NLIDataset(dev_df, tokenizer, args.max_length), batch_size=args.eval_batch_size, collate_fn=collator)
    if mode == 'baseline':
        dataset = NLIDataset(train_df, tokenizer, args.max_length)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    else:
        dataset = ReversalDataset(train_df, tokenizer, args.max_length)
        train_loader = DataLoader(dataset, batch_size=args.reverse_batch_size, shuffle=True, collate_fn=reversal_collator(tokenizer))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID).to(device)
    if device.type == "cuda":
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_f1, best_epoch, patience, history = -float('inf'), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3)
        count = 0
        for batch in tqdm(train_loader, desc=f'{mode} epoch {epoch}/{args.epochs}'):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if mode == 'baseline':
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = model(**batch).loss
                    cls, cons = loss, torch.zeros_like(loss)
                    n = len(batch['labels'])
                else:
                    loss, cls, cons = reversal_loss(model, batch, device, args.consistency_weight)
                    n = len(batch['labels_original'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}; stop and inspect numerical stability')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals += np.array([loss.detach().item(), cls.detach().item(), cons.detach().item()]) * n
            count += n
        metrics = evaluate(model, dev_loader, device)
        row = {'epoch': epoch, 'train_total_loss': totals[0]/count,
               'train_classification_loss': totals[1]/count, 'train_consistency_loss': totals[2]/count, **metrics}
        history.append(row)
        print(json.dumps(row, indent=2))
        if metrics['dev_weighted_f1'] > best_f1 + args.min_delta:
            best_f1, best_epoch, patience = metrics['dev_weighted_f1'], epoch, 0
            model.save_pretrained(out / 'best_model', safe_serialization=True)
            tokenizer.save_pretrained(out / 'best_model')
            print(f'Saved best {mode} model (epoch {epoch})')
        else:
            patience += 1
            if patience >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break
    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    del model, optimizer, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if device.type == 'mps':
        torch.mps.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(out / 'best_model').to(device)
    best_model.eval()
    preds, confidences = [], []
    with torch.inference_mode():
        for batch in tqdm(dev_loader, desc=f'{mode} predictions'):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = best_model(**batch).logits
            probabilities = logits.float().softmax(-1).cpu()
            preds.extend(probabilities.argmax(-1).tolist())
            confidences.extend(probabilities.max(-1).values.tolist())
    prediction_df = dev_df[['instance_id', 'pair_id', 'label']].copy()
    prediction_df['predicted_label'] = [ID2LABEL[p] for p in preds]
    prediction_df['confidence'] = confidences
    prediction_df.to_csv(out / 'dev_predictions.csv', index=False)
    report = classification_report(prediction_df.label, prediction_df.predicted_label, zero_division=0)
    print(f'\n{mode.upper()} best epoch {best_epoch}, weighted F1 {best_f1:.4f}\n{report}')
    (out / 'classification_report.txt').write_text(report)
    pd.DataFrame([{'model': MODEL_NAME, 'training': mode, 'best_epoch': best_epoch,
                   'accuracy': accuracy_score(prediction_df.label, prediction_df.predicted_label),
                   'weighted_f1': f1_score(prediction_df.label, prediction_df.predicted_label, average='weighted', zero_division=0),
                   'macro_f1': f1_score(prediction_df.label, prediction_df.predicted_label, average='macro', zero_division=0)}]).to_csv(out / 'metrics.csv', index=False)
    del best_model
    if device.type == 'mps':
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, default=Path('data/dico_nli_train_track1_participant_labeled.csv'))
    parser.add_argument('--dev-csv', type=Path, default=Path('data/dico_nli_dev_track1_participant_labeled.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--mode', choices=['validate', 'baseline', 'reversal', 'both'], default='validate')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--reverse-batch-size', type=int, default=1)
    parser.add_argument('--eval-batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--consistency-weight', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--min-delta', type=float, default=0.001)
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    train_df, dev_df = load_data(args.train_csv, 'train'), load_data(args.dev_csv, 'dev')
    overlap = set(train_df.pair_id) & set(dev_df.pair_id)
    if overlap:
        raise ValueError(f'Data leakage: {len(overlap)} pair_id values overlap between train and dev')
    train_issues, dev_issues = validate_pairs(train_df, 'train'), validate_pairs(dev_df, 'dev')
    if args.mode == 'validate':
        print('Validation finished; no model downloaded or trained.')
        return
    if train_issues or dev_issues:
        raise ValueError('Invalid directional pairs detected; fix the data before training')
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is not available in this PyTorch installation')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
    print('Using device:', device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in (['baseline', 'reversal'] if args.mode == 'both' else [args.mode]):
        train(args, mode, train_df, dev_df, tokenizer, device)
    print('All requested runs completed. Results:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
