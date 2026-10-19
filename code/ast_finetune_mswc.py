"""
AST Fine-Tuning on MSWC English - Google Colab A100
====================================================
Fine-tunes a pretrained Audio Spectrogram Transformer (MIT/ast-finetuned-audioset)
on 1,000 and 6,210 class MSWC English keyword spotting.

This is the "nuclear option" — pretrained 87M parameter model vs our
from-scratch 2.9M KWT-2 that topped out at 81.2%.

Expected accuracy: 88-93% on 1,000 classes.

Run in Google Colab after downloading and extracting MSWC English.
"""

# =====================================================
# CELL 1: Setup and imports (run after pip install)
# =====================================================

import os
import sys
import csv
import json
import time
import warnings
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import librosa

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ASTModel, ASTFeatureExtractor

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# =====================================================
# PATHS — adjust based on extraction structure
# =====================================================
# After extracting mswc.tar.gz and then audio/en.tar.gz + splits/en.tar.gz
def _data_path(default):
    """Return the directory given after --data_path on the command line, or the default."""
    if "--data_path" in sys.argv:
        i = sys.argv.index("--data_path")
        if i + 1 >= len(sys.argv):
            sys.exit("--data_path needs a directory")
        path = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return path
    return default

# MSWC root containing audio/en/clips and splits/ (pass --data_path)
BASE_DIR = _data_path("mswc")
CLIPS_DIR = os.path.join(BASE_DIR, "audio", "en", "clips")
TRAIN_CSV = os.path.join(BASE_DIR, "splits", "en_train.csv")
TEST_CSV = os.path.join(BASE_DIR, "splits", "en_test.csv")

CONFIG = {
    'sample_rate': 16000,
    'max_audio_len': 16000,  # 1 second at 16kHz
    'random_state': 42,
    # experiment configs
    'experiments': {
        '1000class': {
            'name': '1,000 classes',
            'min_samples': 500,
            'max_train': 300,
            'max_test': 50,
            'top_k': 1000,
        },
        '6210class': {
            'name': '6,210 classes',
            'min_samples': 100,
            'max_train': 50,
            'max_test': 30,
            'top_k': None,
        },
    },
    # training
    'batch_size': 16,
    'lr': 1e-5,
    'weight_decay': 0.01,
    'epochs': 20,
    'warmup_ratio': 0.1,
    'fp16': True,
}


# =====================================================
# DATA LOADING
# =====================================================

def count_words_in_csv(csv_path):
    counts = defaultdict(int)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row['WORD']] += 1
    return counts


def select_keywords(train_counts, min_samples, top_k=None):
    qualified = {kw: c for kw, c in train_counts.items() if c >= min_samples}
    sorted_kws = sorted(qualified.keys(), key=lambda k: -qualified[k])
    if top_k:
        sorted_kws = sorted_kws[:top_k]
    return sorted_kws


def collect_links(csv_path, keywords, max_per_kw, seed=42):
    rng = np.random.default_rng(seed)
    kw_set = set(keywords)
    by_kw = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['WORD'] in kw_set:
                by_kw[row['WORD']].append(row['LINK'])

    kw_to_idx = {kw: i for i, kw in enumerate(keywords)}
    links, labels = [], []
    for kw in keywords:
        avail = by_kw.get(kw, [])
        if not avail:
            continue
        n = min(max_per_kw, len(avail))
        chosen = rng.choice(avail, size=n, replace=False).tolist() if len(avail) > max_per_kw else avail
        links.extend(chosen)
        labels.extend([kw_to_idx[kw]] * len(chosen))
    return links, labels, kw_to_idx


# =====================================================
# AST DATASET — loads audio on-the-fly, uses AST feature extractor
# =====================================================

class MswcASTDataset(Dataset):
    """
    Lazy-loading dataset that:
    1. Loads .opus audio from disk
    2. Pads/truncates to 1 second
    3. Returns raw waveform (AST feature extractor handles mel spectrogram)
    """
    def __init__(self, links, labels, clips_dir, feature_extractor, max_audio_len=16000):
        self.links = links
        self.labels = labels
        self.clips_dir = clips_dir
        self.fe = feature_extractor
        self.max_len = max_audio_len

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        path = os.path.join(self.clips_dir, self.links[idx])
        try:
            audio, sr = librosa.load(path, sr=16000, mono=True)
            if len(audio) == 0:
                audio = np.zeros(self.max_len, dtype=np.float32)
        except:
            audio = np.zeros(self.max_len, dtype=np.float32)

        # pad or truncate to exactly 1 second
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        elif len(audio) < self.max_len:
            audio = np.pad(audio, (0, self.max_len - len(audio)))

        # AST feature extractor converts waveform to mel spectrogram
        inputs = self.fe(
            audio, sampling_rate=16000,
            return_tensors="pt", padding=False
        )
        input_values = inputs['input_values'].squeeze(0)  # (1, time, freq) -> (time, freq)

        return input_values, self.labels[idx]


def collate_fn(batch):
    """Custom collate to handle variable-size spectrograms"""
    inputs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return inputs, labels


# =====================================================
# AST CLASSIFIER MODEL
# =====================================================

class ASTClassifier(nn.Module):
    """
    Pretrained AST encoder + custom classification head.
    Loads MIT/ast-finetuned-audioset weights, replaces the head.
    """
    def __init__(self, n_classes, freeze_encoder_layers=0):
        super().__init__()
        print(f"  Loading pretrained AST from HuggingFace...")
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        hidden_size = self.ast.config.hidden_size  # 768

        # classification head with expansion (softmax bottleneck fix)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes)
        )

        # optionally freeze early encoder layers
        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.ast.encoder.layer):
                if i < freeze_encoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"  Frozen first {freeze_encoder_layers} encoder layers")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  AST Classifier: {total:,} total params | {trainable:,} trainable | {n_classes} classes")

    def forward(self, x):
        # x shape: (batch, time, freq) — AST expects (batch, time, freq)
        outputs = self.ast(x)
        # use CLS token output
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)


# =====================================================
# TRAINING LOOP
# =====================================================

def train_ast(train_loader, test_loader, n_classes, n_test, config, exp_name):
    device = 'cuda'
    print(f"\n{'='*70}")
    print(f"AST Fine-Tuning: {exp_name}")
    print(f"Train batches: {len(train_loader)} | Test samples: {n_test} | Classes: {n_classes}")
    print(f"Batch size: {config['batch_size']} | LR: {config['lr']} | Epochs: {config['epochs']}")
    print(f"FP16: {config['fp16']} | Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")

    # freeze first 8 of 12 encoder layers to save VRAM and speed up training
    model = ASTClassifier(n_classes, freeze_encoder_layers=8).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # cosine schedule
    total_steps = config['epochs'] * len(train_loader)
    warmup_steps = int(total_steps * config['warmup_ratio'])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if config['fp16'] else None

    t_start = time.time()
    best_acc = 0
    best_state = None

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        nb = 0
        epoch_start = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            if config['fp16']:
                with autocast(dtype=torch.float16):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            nb += 1

        avg_loss = total_loss / nb
        epoch_time = time.time() - epoch_start

        # evaluate every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if config['fp16']:
                        with autocast(dtype=torch.float16):
                            logits = model(xb)
                    else:
                        logits = model(xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            marker = " *BEST*" if acc > best_acc else ""
            print(f"  Epoch {epoch+1:>2}/{config['epochs']}: loss={avg_loss:.4f} acc={acc:.4f} lr={lr:.8f} [{epoch_time/60:.1f}m]{marker}")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - t_start
    print(f"  Training complete: {total_time/60:.1f} min | Best acc: {best_acc:.4f}")

    # final eval with best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            if config['fp16']:
                with autocast(dtype=torch.float16):
                    logits = model(xb)
            else:
                logits = model(xb)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'best_val_acc': float(best_acc),
        'train_time_min': float(total_time / 60),
        'n_classes': n_classes,
    }


# =====================================================
# MAIN — run both experiments
# =====================================================

def run_experiment(exp_key, exp_cfg, config):
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {exp_cfg['name']}")
    print(f"{'#'*70}")

    # check paths
    for p, n in [(CLIPS_DIR, "clips"), (TRAIN_CSV, "train CSV"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(p):
            print(f"ERROR: {n} not found at {p}")
            return None

    # count words
    print("Counting words...")
    train_counts = count_words_in_csv(TRAIN_CSV)
    keywords = select_keywords(train_counts, exp_cfg['min_samples'], exp_cfg['top_k'])
    n_classes = len(keywords)
    print(f"  Selected {n_classes} keywords")

    # collect links
    print("Collecting train links...")
    train_links, train_labels, kw_to_idx = collect_links(
        TRAIN_CSV, keywords, exp_cfg['max_train'], seed=42
    )
    print(f"  Train: {len(train_links)} samples")

    print("Collecting test links...")
    test_links, test_labels, _ = collect_links(
        TEST_CSV, keywords, exp_cfg['max_test'], seed=42
    )
    print(f"  Test: {len(test_links)} samples")

    # AST feature extractor
    print("Loading AST feature extractor...")
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # datasets
    train_dataset = MswcASTDataset(train_links, train_labels, CLIPS_DIR, fe)
    test_dataset = MswcASTDataset(test_links, test_labels, CLIPS_DIR, fe)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=True
    )

    # train
    results = train_ast(
        train_loader, test_loader, n_classes,
        len(test_links), config, exp_cfg['name']
    )

    # save keywords for reference
    results['keywords_count'] = n_classes
    results['train_samples'] = len(train_links)
    results['test_samples'] = len(test_links)

    return results


def main():
    print("=" * 70)
    print("AST Fine-Tuning on MSWC English")
    print("Pretrained: MIT/ast-finetuned-audioset-10-10-0.4593 (87M params)")
    print("=" * 70)

    all_results = {}

    for exp_key, exp_cfg in CONFIG['experiments'].items():
        results = run_experiment(exp_key, exp_cfg, CONFIG)
        if results:
            all_results[exp_key] = results

        # free GPU memory between experiments
        torch.cuda.empty_cache()

    # final comparison table
    print("\n" + "=" * 95)
    print("COMPLETE MODEL COMPARISON: MSWC English Keyword Spotting")
    print("=" * 95)
    print(f"{'Model':<50} {'Classes':<10} {'Accuracy':<12} {'Params':<12}")
    print("-" * 85)
    print(f"{'CNN (Lab 8 baseline)':<50} {'31':<10} {'86.7%':<12} {'~150K':<12}")
    print(f"{'KWT-2 v1 (13-bin MFCC, from scratch)':<50} {'1,000':<10} {'77.6%':<12} {'1.7M':<12}")
    print(f"{'KWT-2 v2 (40-bin logmel, from scratch)':<50} {'1,000':<10} {'81.2%':<12} {'2.9M':<12}")
    for exp_key, r in all_results.items():
        name = f"AST (pretrained, fine-tuned) - {CONFIG['experiments'][exp_key]['name']}"
        acc_str = f"{r['accuracy']*100:.1f}%"
        print(f"{name:<50} {str(r['n_classes']):<10} {acc_str:<12} {'87M':<12}")
    print("-" * 85)
    print(f"{'MSWC Paper: DSCNN supervised':<50} {'7':<10} {'85.3%':<12} {'-':<12}")
    print(f"{'MSWC Paper: 5-shot transfer':<50} {'7':<10} {'94.7%':<12} {'-':<12}")

    # save results
    all_results['_meta'] = {
        'model': 'AST (MIT/ast-finetuned-audioset-10-10-0.4593)',
        'pretrained_on': 'AudioSet + ImageNet',
        'fine_tuned_on': 'MSWC English',
        'gpu': torch.cuda.get_device_name(0),
        'fp16': CONFIG['fp16'],
        'frozen_layers': 8,
        'batch_size': CONFIG['batch_size'],
        'lr': CONFIG['lr'],
        'epochs': CONFIG['epochs'],
    }

    with open('ast_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to ast_results.json")
    print("Done!")


if __name__ == "__main__":
    main()
