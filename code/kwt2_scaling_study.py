"""
KWT-2 Vocabulary Scaling Study on Full MSWC English
Precompute MFCCs to disk, then train from memory for fast GPU training.

Runs two experiments:
  Experiment B: 1,000 most common keywords x 500 samples
  Experiment C: 6,210 keywords (100+ samples) x 200 samples

Usage:
    python kwt2_scaling_study.py --data_path /path/to/mswc --precompute   (step 1: extract MFCCs, ~70 min)
    python kwt2_scaling_study.py --data_path /path/to/mswc --train        (step 2: train both scales, ~25 min)
    python kwt2_scaling_study.py --data_path /path/to/mswc --all          (both steps sequentially)
"""

import os
import sys
import csv
import json
import math
import time
import warnings
from collections import defaultdict
import numpy as np
import librosa

warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# =====================================================
# PATHS
# =====================================================
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
CACHE_DIR = os.path.join(BASE_DIR, "mswc_cache")

MFCC_CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400,
    'n_mels': 40,
    'max_len': 100,
}

TRAIN_CONFIG = {
    'd_model': 128,
    'n_heads': 2,
    'n_layers': 12,
    'd_ff': 256,
    'epochs': 30,
    'batch_size': 256,
    'lr': 0.001,
    'weight_decay': 0.1,
    'warmup_epochs': 5,
    'label_smoothing': 0.1,
    'spec_freq_mask_size': 4,
    'spec_time_mask_size': 25,
    'spec_n_masks': 2,
}

# two experiment configs
EXPERIMENTS = {
    'B_1000class': {
        'name': 'Experiment B: 1,000 classes',
        'min_samples': 500,
        'max_train': 500,
        'max_test': 50,
        'top_k': 1000,
    },
    'C_6210class': {
        'name': 'Experiment C: 6,210 classes',
        'min_samples': 100,
        'max_train': 200,
        'max_test': 30,
        'top_k': None,
    },
}


# =====================================================
# MFCC EXTRACTION
# =====================================================

def extract_mfcc(audio, cfg=MFCC_CONFIG):
    audio = audio.astype(np.float32)
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx
    mfcc = librosa.feature.mfcc(
        y=audio, sr=cfg['sample_rate'], n_mfcc=cfg['n_mfcc'],
        n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
        win_length=cfg['win_length'], n_mels=cfg['n_mels']
    )
    if mfcc.shape[1] < cfg['max_len']:
        mfcc = np.pad(mfcc, ((0, 0), (0, cfg['max_len'] - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :cfg['max_len']]
    return mfcc


# =====================================================
# SPECAUGMENT (applied during training on GPU tensors)
# =====================================================

def spec_augment_batch(mfcc_batch, n_masks=2, freq_size=4, time_size=25):
    """apply specaugment to a batch of mfcc tensors on GPU"""
    batch = mfcc_batch.clone()
    B, T, F = batch.shape  # (batch, 100, 13)
    for _ in range(n_masks):
        f = torch.randint(0, max(1, freq_size), (1,)).item()
        f0 = torch.randint(0, max(1, F - f), (1,)).item()
        batch[:, :, f0:f0+f] = 0
    for _ in range(n_masks):
        t = torch.randint(0, max(1, time_size), (1,)).item()
        t0 = torch.randint(0, max(1, T - t), (1,)).item()
        batch[:, t0:t0+t, :] = 0
    return batch


# =====================================================
# BUILD WORD COUNTS FROM CSV
# =====================================================

def count_words_in_csv(csv_path):
    """count samples per keyword in a csv file"""
    counts = defaultdict(int)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row['WORD']] += 1
    return counts


def select_keywords(train_counts, min_samples, top_k=None):
    """select keywords that meet minimum sample threshold"""
    qualified = {kw: c for kw, c in train_counts.items() if c >= min_samples}
    sorted_kws = sorted(qualified.keys(), key=lambda k: -qualified[k])
    if top_k is not None:
        sorted_kws = sorted_kws[:top_k]
    return sorted_kws


def collect_links(csv_path, keywords, max_per_kw, seed=42):
    """collect balanced file links for selected keywords"""
    rng = np.random.default_rng(seed)
    kw_set = set(keywords)
    by_kw = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row['WORD']
            if w in kw_set:
                by_kw[w].append(row['LINK'])

    kw_to_idx = {kw: i for i, kw in enumerate(keywords)}
    links = []
    labels = []
    for kw in keywords:
        available = by_kw.get(kw, [])
        if len(available) == 0:
            continue
        n = min(max_per_kw, len(available))
        if len(available) > max_per_kw:
            chosen = rng.choice(available, size=n, replace=False).tolist()
        else:
            chosen = available
        links.extend(chosen)
        labels.extend([kw_to_idx[kw]] * len(chosen))
    return links, np.array(labels), kw_to_idx


# =====================================================
# PRECOMPUTE MFCCS
# =====================================================

def precompute_mfccs(links, labels, clips_dir, save_path, desc="dataset"):
    """load all audio, extract mfcc, save as single .pt file"""
    total = len(links)
    mfccs = []
    valid_labels = []
    failed = 0

    print(f"  Precomputing MFCCs for {total} files ({desc})...")
    t0 = time.time()

    for i, (link, label) in enumerate(zip(links, labels)):
        path = os.path.join(clips_dir, link)
        try:
            audio, _ = librosa.load(path, sr=MFCC_CONFIG['sample_rate'], mono=True)
            if len(audio) == 0:
                failed += 1
                continue
            mfcc = extract_mfcc(audio)
            # transpose to (100, 13) for transformer
            mfccs.append(mfcc.T)
            valid_labels.append(label)
        except Exception:
            failed += 1
            continue

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"    {i+1}/{total} ({failed} failed) - {rate:.0f} files/sec - ~{remaining/60:.1f} min remaining")

    X = torch.FloatTensor(np.array(mfccs))
    y = torch.LongTensor(valid_labels)

    print(f"  Done: {len(mfccs)} loaded, {failed} failed, shape={X.shape}")
    print(f"  Size in memory: {X.element_size() * X.nelement() / 1e9:.2f} GB")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    torch.save({'X': X, 'y': y}, save_path)
    print(f"  Saved to {save_path}")
    return X, y


# =====================================================
# KWT-2 MODEL
# =====================================================

class KWT2(nn.Module):
    def __init__(self, n_mfcc=13, seq_len=100, d_model=128, n_heads=2,
                 n_layers=12, d_ff=256, n_classes=1000):
        super().__init__()
        self.input_proj = nn.Linear(n_mfcc, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            activation='gelu', batch_first=True, norm_first=False, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        total = sum(p.numel() for p in self.parameters())
        print(f"  KWT-2: {total:,} params | {n_classes} classes | d={d_model} h={n_heads} L={n_layers}")

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(self.norm(x[:, 0]))


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            p = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * p))
        for g in self.optimizer.param_groups:
            g['lr'] = lr


# =====================================================
# TRAIN FROM PRECOMPUTED TENSORS
# =====================================================

def train_from_cache(X_train, y_train, X_test, y_test, n_classes, exp_name, cfg=TRAIN_CONFIG):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"Training: {exp_name}")
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Classes: {n_classes} | Device: {device}")
    print(f"{'='*70}")

    # move test set to GPU (small enough)
    X_te = X_test.to(device)
    y_te = y_test.to(device)

    # train set stays on CPU, batches moved to GPU
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=True)

    model = KWT2(
        n_mfcc=MFCC_CONFIG['n_mfcc'], seq_len=MFCC_CONFIG['max_len'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'], n_classes=n_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    spe = len(train_loader)
    scheduler = CosineWarmupScheduler(optimizer, cfg['warmup_epochs'] * spe, cfg['epochs'] * spe)

    t_start = time.time()
    best_acc = 0
    best_state = None

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0
        nb = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # specaugment on GPU
            xb = spec_augment_batch(xb, cfg['spec_n_masks'],
                                    cfg['spec_freq_mask_size'],
                                    cfg['spec_time_mask_size'])
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            nb += 1

        avg_loss = total_loss / nb

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # evaluate in chunks to avoid OOM
                correct = 0
                total = 0
                chunk = 512
                for i in range(0, len(X_te), chunk):
                    logits = model(X_te[i:i+chunk])
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_te[i:i+chunk]).sum().item()
                    total += min(chunk, len(X_te) - i)
                acc = correct / total

            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:>2}/{cfg['epochs']}: loss={avg_loss:.4f} acc={acc:.4f} lr={lr:.6f} [{elapsed/60:.1f}m]")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - t_start
    print(f"  Training complete: {total_time/60:.1f} min | Best acc: {best_acc:.4f}")

    # final eval with best model
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        all_preds = []
        chunk = 512
        for i in range(0, len(X_te), chunk):
            preds = model(X_te[i:i+chunk]).argmax(dim=1).cpu()
            all_preds.append(preds)
    y_pred = torch.cat(all_preds).numpy()
    y_true = y_test.numpy()

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
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
    }


# =====================================================
# PRECOMPUTE STEP
# =====================================================

def run_precompute():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Counting words in training CSV...")
    train_counts = count_words_in_csv(TRAIN_CSV)
    print(f"  Total unique words: {len(train_counts)}")

    for exp_key, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'='*70}")
        print(f"PRECOMPUTING: {exp_cfg['name']}")
        print(f"{'='*70}")

        keywords = select_keywords(train_counts, exp_cfg['min_samples'], exp_cfg['top_k'])
        n_classes = len(keywords)
        print(f"  Selected {n_classes} keywords")

        # save keyword list
        kw_path = os.path.join(CACHE_DIR, f"{exp_key}_keywords.json")
        with open(kw_path, 'w') as f:
            json.dump(keywords, f)

        # train
        print(f"\n  --- TRAIN SET ---")
        train_links, train_labels, kw_to_idx = collect_links(
            TRAIN_CSV, keywords, exp_cfg['max_train'], seed=42
        )
        train_cache = os.path.join(CACHE_DIR, f"{exp_key}_train.pt")
        precompute_mfccs(train_links, train_labels, CLIPS_DIR, train_cache, "train")

        # test
        print(f"\n  --- TEST SET ---")
        test_links, test_labels, _ = collect_links(
            TEST_CSV, keywords, exp_cfg['max_test'], seed=42
        )
        test_cache = os.path.join(CACHE_DIR, f"{exp_key}_test.pt")
        precompute_mfccs(test_links, test_labels, CLIPS_DIR, test_cache, "test")

    print(f"\nPrecomputation complete! Cache files in {CACHE_DIR}")


# =====================================================
# TRAIN STEP
# =====================================================

def run_train():
    all_results = {}

    for exp_key, exp_cfg in EXPERIMENTS.items():
        print(f"\nLoading cache for {exp_cfg['name']}...")

        kw_path = os.path.join(CACHE_DIR, f"{exp_key}_keywords.json")
        train_path = os.path.join(CACHE_DIR, f"{exp_key}_train.pt")
        test_path = os.path.join(CACHE_DIR, f"{exp_key}_test.pt")

        for p in [kw_path, train_path, test_path]:
            if not os.path.exists(p):
                print(f"  ERROR: {p} not found. Run --precompute first.")
                return

        with open(kw_path) as f:
            keywords = json.load(f)

        train_data = torch.load(train_path, weights_only=True)
        test_data = torch.load(test_path, weights_only=True)

        X_train, y_train = train_data['X'], train_data['y']
        X_test, y_test = test_data['X'], test_data['y']

        print(f"  Train: {X_train.shape} | Test: {X_test.shape} | Classes: {len(keywords)}")

        results = train_from_cache(X_train, y_train, X_test, y_test, len(keywords), exp_cfg['name'])
        all_results[exp_key] = results

        # free memory
        del X_train, y_train, X_test, y_test, train_data, test_data
        torch.cuda.empty_cache()

    # final comparison
    print("\n" + "=" * 90)
    print("VOCABULARY SCALING STUDY: KWT-2 on MSWC English")
    print("=" * 90)
    print(f"{'Experiment':<45} {'Classes':<10} {'Train':<10} {'Test':<10} {'Accuracy':<12}")
    print("-" * 90)
    # microset baseline
    print(f"{'Microset CNN (baseline)':<45} {'31':<10} {'17,923':<10} {'4,553':<10} {'86.7%':<12}")
    print(f"{'Microset KWT-2':<45} {'31':<10} {'17,923':<10} {'4,553':<10} {'86.8%':<12}")

    for exp_key, r in all_results.items():
        name = EXPERIMENTS[exp_key]['name']
        acc_str = f"{r['accuracy']*100:.1f}%"
        print(f"{'Full ' + name:<45} {str(r['n_classes']):<10} {str(r['train_samples']):<10} {str(r['test_samples']):<10} {acc_str:<12}")

    # paper baselines
    print("-" * 90)
    print(f"{'MSWC Paper: DSCNN supervised':<45} {'7':<10} {'-':<10} {'-':<10} {'85.3%':<12}")
    print(f"{'MSWC Paper: 5-shot transfer':<45} {'7':<10} {'-':<10} {'-':<10} {'94.7%':<12}")

    # save
    all_results['_meta'] = {
        'architecture': 'KWT-2 (Berg et al., Interspeech 2021)',
        'gpu': 'NVIDIA GeForce RTX 3060 12GB',
        'augmentation': 'SpecAugment (on-GPU)',
        'optimizer': f"AdamW lr={TRAIN_CONFIG['lr']} wd={TRAIN_CONFIG['weight_decay']}",
    }
    with open("kwt2_scaling_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to kwt2_scaling_results.json")
    print("Done!")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python kwt2_scaling_study.py --data_path /path/to/mswc --precompute   (extract MFCCs, ~70 min)")
        print("  python kwt2_scaling_study.py --data_path /path/to/mswc --train        (train both experiments, ~25 min)")
        print("  python kwt2_scaling_study.py --data_path /path/to/mswc --all          (both steps)")
        sys.exit(1)

    mode = sys.argv[1]

    if mode in ('--precompute', '--all'):
        run_precompute()

    if mode in ('--train', '--all'):
        run_train()
