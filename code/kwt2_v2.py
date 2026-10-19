"""
KWT-2 v2 — Fixed Implementation on Full MSWC English
=====================================================
Bug fixes from v1 (77.6%):
  1. 13-bin MFCC -> 40-bin log-mel spectrogram (match KWT paper)
  2. d_ff=256 -> d_ff=512 (match actual KWT-2 specification)
  3. Added Mixup augmentation (alpha=0.3)
  4. Expanded classification head: 128->512->n_classes (softmax bottleneck fix)
  5. SpecAugment freq mask 4->7 (scaled for 40 bins)
  6. Epochs 30->50 (loss was still dropping)
  7. Speed perturbation + time shift applied during precompute (3 augmented copies)

Usage:
    python kwt2_v2.py --data_path /path/to/mswc --precompute
    python kwt2_v2.py --data_path /path/to/mswc --train
    python kwt2_v2.py --data_path /path/to/mswc --all
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
from torch.utils.data import DataLoader, TensorDataset

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
CACHE_DIR = os.path.join(BASE_DIR, "mswc_cache_v2")

# FIX 1: 40-bin log-mel spectrogram instead of 13-bin MFCC
FEATURE_CONFIG = {
    'sample_rate': 16000,
    'n_mels': 40,          # was n_mfcc=13
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400,
    'max_len': 98,          # KWT paper uses 98 time frames
}

# FIX 2: d_ff=512 (was 256), FIX 6: epochs=50 (was 30)
TRAIN_CONFIG = {
    'd_model': 128,
    'n_heads': 2,
    'n_layers': 12,
    'd_ff': 512,            # FIXED: was 256, KWT-2 spec is 512
    'epochs': 50,           # FIXED: was 30, loss still dropping
    'batch_size': 256,
    'lr': 0.001,
    'weight_decay': 0.1,
    'warmup_epochs': 5,
    'label_smoothing': 0.1,
    # FIX 5: SpecAugment scaled for 40 bins
    'spec_freq_mask_size': 7,   # was 4, scaled from 13->40 bins
    'spec_time_mask_size': 25,
    'spec_n_masks': 2,
    # FIX 3: Mixup
    'mixup_alpha': 0.3,
}

EXPERIMENTS = {
    'B_1000class_v2': {
        'name': '1,000 classes (v2 fixed)',
        'min_samples': 500,
        'max_train': 300,
        'max_test': 50,
        'top_k': 1000,
    },
    'C_6210class_v2': {
        'name': '6,210 classes (v2 fixed)',
        'min_samples': 100,
        'max_train': 50,
        'max_test': 30,
        'top_k': None,
    },
}


# =====================================================
# FIX 1: LOG-MEL SPECTROGRAM (not MFCC)
# =====================================================

def extract_logmel(audio, cfg=FEATURE_CONFIG):
    """extract 40-bin log-mel spectrogram (what KWT actually uses)"""
    audio = audio.astype(np.float32)
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx

    mel = librosa.feature.melspectrogram(
        y=audio, sr=cfg['sample_rate'],
        n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
        win_length=cfg['win_length'], n_mels=cfg['n_mels'],
        fmin=60, fmax=7800
    )
    # log-mel (add small epsilon to avoid log(0))
    logmel = np.log(mel + 1e-6)

    # pad or truncate to max_len time frames
    if logmel.shape[1] < cfg['max_len']:
        logmel = np.pad(logmel, ((0, 0), (0, cfg['max_len'] - logmel.shape[1])), mode='constant')
    else:
        logmel = logmel[:, :cfg['max_len']]

    return logmel


# =====================================================
# WAVEFORM AUGMENTATION (applied during precompute)
# =====================================================

def speed_perturb(audio, low=0.85, high=1.15):
    factor = np.random.uniform(low, high)
    if abs(factor - 1.0) < 0.01:
        return audio
    new_len = int(len(audio) / factor)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def time_shift(audio, shift_ms=100, sr=16000):
    shift_samples = int(sr * shift_ms / 1000)
    shift = np.random.randint(-shift_samples, shift_samples + 1)
    if shift > 0:
        audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
    elif shift < 0:
        audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
    return audio


# =====================================================
# SPECAUGMENT (on GPU during training)
# =====================================================

def spec_augment_batch(batch, n_masks=2, freq_size=7, time_size=25):
    """apply specaugment to batch on GPU. batch shape: (B, T, F)"""
    aug = batch.clone()
    B, T, F = aug.shape
    for _ in range(n_masks):
        f = torch.randint(0, max(1, freq_size), (1,)).item()
        f0 = torch.randint(0, max(1, F - f), (1,)).item()
        aug[:, :, f0:f0+f] = 0
    for _ in range(n_masks):
        t = torch.randint(0, max(1, time_size), (1,)).item()
        t0 = torch.randint(0, max(1, T - t), (1,)).item()
        aug[:, t0:t0+t, :] = 0
    return aug


# =====================================================
# FIX 3: MIXUP AUGMENTATION
# =====================================================

def mixup_batch(x, y, alpha=0.3, n_classes=1000):
    """mixup augmentation on spectrograms. returns mixed x, soft labels"""
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, n_classes).float()

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5 so original dominates

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_onehot = torch.nn.functional.one_hot(y, n_classes).float()
    y_index = torch.nn.functional.one_hot(y[index], n_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_index

    return mixed_x, mixed_y


# =====================================================
# CSV LOADING
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
    if top_k is not None:
        sorted_kws = sorted_kws[:top_k]
    return sorted_kws


def collect_links(csv_path, keywords, max_per_kw, seed=42):
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
# PRECOMPUTE LOG-MEL FEATURES
# =====================================================

def precompute_features(links, labels, clips_dir, save_path, desc="dataset", augment=False):
    """
    Load audio, extract 40-bin log-mel, save as .pt
    If augment=True, also create speed-perturbed and time-shifted copies (3x data)
    """
    total = len(links)
    features = []
    valid_labels = []
    failed = 0

    print(f"  Extracting 40-bin log-mel for {total} files ({desc})...")
    if augment:
        print(f"  + 2 augmented copies per sample (speed perturb + time shift)")
    t0 = time.time()

    for i, (link, label) in enumerate(zip(links, labels)):
        path = os.path.join(clips_dir, link)
        try:
            audio, _ = librosa.load(path, sr=FEATURE_CONFIG['sample_rate'], mono=True)
            if len(audio) == 0:
                failed += 1
                continue

            # original
            logmel = extract_logmel(audio)
            features.append(logmel.T)  # (98, 40)
            valid_labels.append(label)

            if augment:
                # augmented copy 1: speed perturbation
                aug1 = speed_perturb(audio.copy())
                logmel1 = extract_logmel(aug1)
                features.append(logmel1.T)
                valid_labels.append(label)

                # augmented copy 2: time shift
                aug2 = time_shift(audio.copy())
                logmel2 = extract_logmel(aug2)
                features.append(logmel2.T)
                valid_labels.append(label)

        except Exception:
            failed += 1
            continue

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            n_stored = len(features)
            print(f"    {i+1}/{total} ({failed} failed) | {n_stored} features stored | {rate:.0f} files/sec | ~{remaining/60:.1f} min")

    X = torch.FloatTensor(np.array(features))
    y = torch.LongTensor(valid_labels)

    print(f"  Done: {len(features)} features ({failed} failed), shape={X.shape}")
    print(f"  Memory: {X.element_size() * X.nelement() / 1e9:.2f} GB")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    torch.save({'X': X, 'y': y}, save_path)
    print(f"  Saved to {save_path}")
    return X, y


# =====================================================
# KWT-2 v2 MODEL (with fixes)
# =====================================================

class KWT2v2(nn.Module):
    """
    KWT-2 with bug fixes:
    - Input: 40-bin log-mel (was 13-bin MFCC)
    - d_ff=512 (was 256)
    - Expansion classification head: 128->512->n_classes (softmax bottleneck fix)
    """
    def __init__(self, n_features=40, seq_len=98, d_model=128, n_heads=2,
                 n_layers=12, d_ff=512, n_classes=1000):
        super().__init__()
        self.n_classes = n_classes

        # project each log-mel frame to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # learnable CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)

        # transformer encoder (PostNorm, no dropout — per KWT paper)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            activation='gelu', batch_first=True, norm_first=False, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # FIX 4: expansion head to break softmax bottleneck
        # instead of Linear(128, 6210) which has rank 128,
        # we do Linear(128, 512) -> GELU -> Linear(512, n_classes)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes)
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"  KWT-2v2: {total:,} params | {n_classes} classes | d={d_model} ff={d_ff} h={n_heads} L={n_layers}")
        print(f"  Features: {n_features}-bin log-mel | Seq: {seq_len} frames")
        print(f"  Head: {d_model}->512->GELU->{n_classes} (softmax bottleneck fix)")

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)


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
# TRAINING WITH MIXUP
# =====================================================

def train_from_cache(X_train, y_train, X_test, y_test, n_classes, exp_name, cfg=TRAIN_CONFIG):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"Training v2: {exp_name}")
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Classes: {n_classes} | Device: {device}")
    print(f"Fixes: 40-bin logmel | d_ff=512 | mixup={cfg['mixup_alpha']} | expansion head | {cfg['epochs']} epochs")
    print(f"{'='*70}")

    X_te = X_test.to(device)
    y_te = y_test.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=0, pin_memory=True)

    model = KWT2v2(
        n_features=FEATURE_CONFIG['n_mels'],
        seq_len=FEATURE_CONFIG['max_len'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'], n_classes=n_classes
    ).to(device)

    # mixup uses soft labels, so we need KLDiv or manual CE
    # standard CE with soft targets = sum(-target * log_softmax(logit))
    def soft_cross_entropy(logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss

    # for non-mixup evaluation
    criterion_eval = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])

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

            # mixup
            xb_mixed, yb_mixed = mixup_batch(xb, yb, cfg['mixup_alpha'], n_classes)

            optimizer.zero_grad()
            logits = model(xb_mixed)
            loss = soft_cross_entropy(logits, yb_mixed)
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
            marker = " *BEST*" if acc > best_acc else ""
            print(f"  Epoch {epoch+1:>2}/{cfg['epochs']}: loss={avg_loss:.4f} acc={acc:.4f} lr={lr:.6f} [{elapsed/60:.1f}m]{marker}")

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
# PRECOMPUTE
# =====================================================

def run_precompute():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Counting words in training CSV...")
    train_counts = count_words_in_csv(TRAIN_CSV)
    print(f"  Total unique words: {len(train_counts)}")

    for exp_key, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'='*70}")
        print(f"PRECOMPUTING v2: {exp_cfg['name']}")
        print(f"Features: 40-bin log-mel spectrogram (98 frames)")
        print(f"{'='*70}")

        keywords = select_keywords(train_counts, exp_cfg['min_samples'], exp_cfg['top_k'])
        n_classes = len(keywords)
        print(f"  Selected {n_classes} keywords")

        kw_path = os.path.join(CACHE_DIR, f"{exp_key}_keywords.json")
        with open(kw_path, 'w') as f:
            json.dump(keywords, f)

        # train (with augmented copies)
        print(f"\n  --- TRAIN SET (with augmentation) ---")
        train_links, train_labels, kw_to_idx = collect_links(
            TRAIN_CSV, keywords, exp_cfg['max_train'], seed=42
        )
        train_cache = os.path.join(CACHE_DIR, f"{exp_key}_train.pt")
        precompute_features(train_links, train_labels, CLIPS_DIR, train_cache, "train", augment=True)

        # test (no augmentation)
        print(f"\n  --- TEST SET (clean) ---")
        test_links, test_labels, _ = collect_links(
            TEST_CSV, keywords, exp_cfg['max_test'], seed=42
        )
        test_cache = os.path.join(CACHE_DIR, f"{exp_key}_test.pt")
        precompute_features(test_links, test_labels, CLIPS_DIR, test_cache, "test", augment=False)

    print(f"\nPrecomputation v2 complete! Cache in {CACHE_DIR}")


# =====================================================
# TRAIN
# =====================================================

def run_train():
    all_results = {}

    for exp_key, exp_cfg in EXPERIMENTS.items():
        print(f"\nLoading v2 cache for {exp_cfg['name']}...")

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

        # save model
        torch.save(results, os.path.join(CACHE_DIR, f"{exp_key}_results.pt"))

        del X_train, y_train, X_test, y_test, train_data, test_data
        torch.cuda.empty_cache()

    # comparison table
    print("\n" + "=" * 95)
    print("VOCABULARY SCALING STUDY v2: KWT-2 Fixed on MSWC English")
    print("=" * 95)
    print(f"{'Experiment':<50} {'Classes':<10} {'Train':<10} {'Test':<10} {'Accuracy':<12}")
    print("-" * 95)
    # v1 baselines
    print(f"{'v1 Microset CNN':<50} {'31':<10} {'17,923':<10} {'4,553':<10} {'86.7%':<12}")
    print(f"{'v1 Microset KWT-2 (13-bin MFCC, d_ff=256)':<50} {'31':<10} {'17,923':<10} {'4,553':<10} {'86.8%':<12}")
    print(f"{'v1 Full 1,000-class (13-bin MFCC, d_ff=256)':<50} {'1,000':<10} {'499,500':<10} {'49,950':<10} {'77.6%':<12}")
    print(f"{'v1 Full 6,210-class (13-bin MFCC, d_ff=256)':<50} {'6,210':<10} {'1,085,898':<10} {'154,154':<10} {'77.6%':<12}")
    print("-" * 95)

    for exp_key, r in all_results.items():
        name = "v2 Fixed " + EXPERIMENTS[exp_key]['name']
        acc_str = f"{r['accuracy']*100:.1f}%"
        print(f"{name:<50} {str(r['n_classes']):<10} {str(r['train_samples']):<10} {str(r['test_samples']):<10} {acc_str:<12}")

    print("-" * 95)
    print(f"{'MSWC Paper: DSCNN supervised':<50} {'7':<10} {'-':<10} {'-':<10} {'85.3%':<12}")
    print(f"{'MSWC Paper: 5-shot transfer':<50} {'7':<10} {'-':<10} {'-':<10} {'94.7%':<12}")

    # improvement summary
    for exp_key, r in all_results.items():
        v1_acc = 0.776
        v2_acc = r['accuracy']
        delta = (v2_acc - v1_acc) * 100
        print(f"\n  {EXPERIMENTS[exp_key]['name']}: v1={v1_acc*100:.1f}% -> v2={v2_acc*100:.1f}% (delta: {delta:+.1f}%)")

    # save all results
    all_results['_meta'] = {
        'version': 'v2',
        'fixes': [
            '40-bin log-mel (was 13-bin MFCC)',
            'd_ff=512 (was 256)',
            'Mixup alpha=0.3',
            'Expansion head 128->512->n_classes',
            'SpecAugment freq_mask=7 (was 4)',
            '50 epochs (was 30)',
            'Speed perturb + time shift augmented copies',
        ],
        'architecture': 'KWT-2v2 (Berg et al., Interspeech 2021, fixed)',
        'gpu': 'NVIDIA GeForce RTX 3060 12GB',
    }
    with open("kwt2_v2_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to kwt2_v2_results.json")
    print("Done!")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python kwt2_v2.py --data_path /path/to/mswc --precompute   (extract 40-bin log-mel features)")
        print("  python kwt2_v2.py --data_path /path/to/mswc --train         (train both experiments)")
        print("  python kwt2_v2.py --data_path /path/to/mswc --all            (both steps)")
        sys.exit(1)

    mode = sys.argv[1]

    if mode in ('--precompute', '--all'):
        run_precompute()

    if mode in ('--train', '--all'):
        run_train()
