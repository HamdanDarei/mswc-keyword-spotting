"""
Keyword Transformer (KWT-2) for Keyword Spotting on MSWC Microset

Architecture: Berg et al., Interspeech 2021
"Keyword Transformer: A Self-Attention Model for Keyword Spotting"

KWT-2 config: 12 layers, 2 heads, d_model=128, ~2.4M parameters
Input: 13-bin MFCC, 100 time frames, each frame = one token
Training: AdamW, cosine schedule, SpecAugment, label smoothing

Baseline to beat: CNN 86.7% on 31-class MSWC microset
Target: approach 94.7% (MSWC paper 5-shot baseline)

Usage:
    source ~/mswc_env/bin/activate
    cd ~/Downloads
    python keyword_spotting_kwt2.py
"""

import os
import sys
import csv
import json
import math
import time
import warnings
from collections import defaultdict, Counter
import numpy as np
import librosa

warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# config
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

# 31-class microset folder containing clips/, en_train.csv, en_test.csv (pass --data_path)
MICROSET_PATH = _data_path("mswc_microset/en")
CLIPS_DIR = os.path.join(MICROSET_PATH, "clips")
TRAIN_CSV = os.path.join(MICROSET_PATH, "en_train.csv")
TEST_CSV = os.path.join(MICROSET_PATH, "en_test.csv")

TARGET_KEYWORDS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'off', 'one', 'right', 'seven', 'sheila', 'six',
    'stop', 'three', 'tree', 'two', 'visual', 'wow', 'yes', 'zero'
]

CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400,
    'n_mels': 40,
    'max_len': 100,
    'random_state': 42,
    'samples_per_keyword_train': 700,
    'samples_per_keyword_test': 200,
    # kwt-2 architecture
    'd_model': 128,
    'n_heads': 2,
    'n_layers': 12,
    'd_ff': 256,
    'n_classes': 31,
    # training
    'epochs': 30,
    'batch_size': 64,
    'lr': 0.001,
    'weight_decay': 0.1,
    'warmup_epochs': 10,
    'label_smoothing': 0.1,
    # augmentation
    'spec_freq_mask_size': 4,
    'spec_time_mask_size': 25,
    'spec_n_masks': 2,
    'speed_perturb_range': (0.85, 1.15),
    'time_shift_ms': 100,
}


# =====================================================
# DATA AUGMENTATION
# =====================================================

def speed_perturb(audio, sr, low=0.85, high=1.15):
    """randomly change speed of audio"""
    factor = np.random.uniform(low, high)
    if abs(factor - 1.0) < 0.01:
        return audio
    # resample to change speed
    new_len = int(len(audio) / factor)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def time_shift(audio, shift_ms, sr):
    """randomly shift audio left or right"""
    shift_samples = int(sr * shift_ms / 1000)
    shift = np.random.randint(-shift_samples, shift_samples + 1)
    if shift > 0:
        audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
    elif shift < 0:
        audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
    return audio


def spec_augment(mfcc, n_freq_masks=2, freq_mask_size=4, n_time_masks=2, time_mask_size=25):
    """apply specaugment masking to mfcc features"""
    augmented = mfcc.copy()
    n_freq, n_time = augmented.shape

    # frequency masks
    for _ in range(n_freq_masks):
        f = np.random.randint(0, max(1, freq_mask_size))
        f0 = np.random.randint(0, max(1, n_freq - f))
        augmented[f0:f0 + f, :] = 0

    # time masks
    for _ in range(n_time_masks):
        t = np.random.randint(0, max(1, time_mask_size))
        t0 = np.random.randint(0, max(1, n_time - t))
        augmented[:, t0:t0 + t] = 0

    return augmented


# =====================================================
# MFCC EXTRACTION
# =====================================================

def extract_mfcc(audio, pad_to=100):
    """extract mfcc features from audio waveform"""
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=CONFIG['sample_rate'],
        n_mfcc=CONFIG['n_mfcc'],
        n_fft=CONFIG['n_fft'],
        hop_length=CONFIG['hop_length'],
        win_length=CONFIG['win_length'],
        n_mels=CONFIG['n_mels']
    )

    if mfcc.shape[1] < pad_to:
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_to - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :pad_to]

    return mfcc


# =====================================================
# DATASET
# =====================================================

class MswcDataset(Dataset):
    """pytorch dataset for mswc audio with on-the-fly augmentation"""

    def __init__(self, links, labels, clips_dir, augment=False):
        self.links = links
        self.labels = labels
        self.clips_dir = clips_dir
        self.augment = augment
        # preload all audio and extract mfcc to avoid repeated disk io
        self.mfccs = []
        self.audios = []
        self.valid_indices = []

        print(f"  Loading {len(links)} audio files...")
        failed = 0
        for i, link in enumerate(links):
            path = os.path.join(clips_dir, link)
            try:
                audio, _ = librosa.load(path, sr=CONFIG['sample_rate'], mono=True)
                if len(audio) == 0:
                    failed += 1
                    continue
                self.audios.append(audio)
                self.valid_indices.append(i)
            except Exception:
                failed += 1
                continue

            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(links)} loaded ({failed} failed)")

        print(f"  Done: {len(self.audios)} loaded, {failed} failed")

        # precompute mfcc for non-augmented use (test set)
        if not augment:
            for audio in self.audios:
                self.mfccs.append(extract_mfcc(audio, CONFIG['max_len']))

        self.valid_labels = [labels[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        label = self.valid_labels[idx]

        if self.augment:
            # apply waveform augmentations
            audio = self.audios[idx].copy()
            audio = speed_perturb(audio, CONFIG['sample_rate'],
                                  CONFIG['speed_perturb_range'][0],
                                  CONFIG['speed_perturb_range'][1])
            audio = time_shift(audio, CONFIG['time_shift_ms'], CONFIG['sample_rate'])
            mfcc = extract_mfcc(audio, CONFIG['max_len'])
            # apply specaugment
            mfcc = spec_augment(mfcc,
                                CONFIG['spec_n_masks'], CONFIG['spec_freq_mask_size'],
                                CONFIG['spec_n_masks'], CONFIG['spec_time_mask_size'])
        else:
            mfcc = self.mfccs[idx]

        # mfcc shape: (13, 100) -> transpose to (100, 13) for transformer (seq_len, features)
        mfcc = mfcc.T  # now (100, 13)

        return torch.FloatTensor(mfcc), torch.LongTensor([label])[0]


# =====================================================
# KWT-2 MODEL
# =====================================================

class KWT2(nn.Module):
    """
    Keyword Transformer (KWT-2)
    Berg et al., Interspeech 2021

    Each time frame of the MFCC is one token.
    Input: (batch, 100, 13) -> project to (batch, 100, 128)
    Prepend learnable CLS token -> (batch, 101, 128)
    12 transformer encoder layers with PostNorm
    Classify from CLS token output
    """

    def __init__(self, n_mfcc=13, seq_len=100, d_model=128, n_heads=2,
                 n_layers=12, d_ff=256, n_classes=31):
        super(KWT2, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # linear projection of each mfcc frame to d_model
        self.input_proj = nn.Linear(n_mfcc, d_model)

        # learnable cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # learnable positional embedding (101 = 100 frames + 1 cls)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)

        # transformer encoder with PostNorm (default in pytorch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            activation='gelu',
            batch_first=True,
            norm_first=False,  # PostNorm (better for KWS per paper)
            dropout=0.0  # no dropout per KWT paper
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        # count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  KWT-2 initialized: {total_params:,} parameters")

    def forward(self, x):
        # x shape: (batch, 100, 13)
        batch_size = x.size(0)

        # project each frame: (batch, 100, 13) -> (batch, 100, 128)
        x = self.input_proj(x)

        # prepend cls token: (batch, 101, 128)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # add positional embedding
        x = x + self.pos_embed

        # transformer encoder
        x = self.transformer(x)

        # take cls token output
        cls_out = x[:, 0]

        # classify
        cls_out = self.norm(cls_out)
        logits = self.head(cls_out)

        return logits


# =====================================================
# COSINE SCHEDULE WITH WARMUP
# =====================================================

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # linear warmup
            lr = self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            # cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for g in self.optimizer.param_groups:
            g['lr'] = lr


# =====================================================
# DATA LOADING
# =====================================================

def load_split_csv(csv_path, target_keywords, samples_per_keyword, seed=42):
    """load file paths from split csv, filter, balance per class"""
    rng = np.random.default_rng(seed)
    target_set = set(target_keywords)
    by_keyword = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['WORD']
            if word in target_set:
                by_keyword[word].append(row['LINK'])

    print(f"  Samples per keyword in {os.path.basename(csv_path)}:")
    for kw in target_keywords:
        n = len(by_keyword[kw])
        marker = " (capped)" if n > samples_per_keyword else ""
        print(f"    {kw:<10} {n:>5}{marker}")

    balanced_links = []
    balanced_labels = []
    kw_to_idx = {kw: i for i, kw in enumerate(target_keywords)}

    for kw in target_keywords:
        available = by_keyword[kw]
        if len(available) == 0:
            print(f"  WARNING: zero samples for '{kw}'")
            continue
        n_take = min(samples_per_keyword, len(available))
        if len(available) >= samples_per_keyword:
            chosen = rng.choice(available, size=n_take, replace=False).tolist()
        else:
            chosen = available
        balanced_links.extend(chosen)
        balanced_labels.extend([kw_to_idx[kw]] * len(chosen))

    return balanced_links, np.array(balanced_labels), kw_to_idx


# =====================================================
# TRAINING
# =====================================================

def train_kwt2(train_dataset, test_dataset, n_classes, config):
    """train kwt-2 and return predictions"""

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"\nTraining KWT-2 on {device}...")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=0, pin_memory=False)

    model = KWT2(
        n_mfcc=config['n_mfcc'],
        seq_len=config['max_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        n_classes=n_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr'],
                            weight_decay=config['weight_decay'])

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['epochs'],
        steps_per_epoch=len(train_loader)
    )

    train_start = time.time()
    best_acc = 0
    best_preds = None

    for epoch in range(config['epochs']):
        # train
        model.train()
        total_loss = 0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    preds = model(xb).argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy())

            acc = accuracy_score(all_labels, all_preds)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}, val_acc={acc:.4f}, lr={current_lr:.6f}")

            if acc > best_acc:
                best_acc = acc
                best_preds = np.array(all_preds)

    train_time = time.time() - train_start
    print(f"  Total training time: {train_time:.1f}s")
    print(f"  Best validation accuracy: {best_acc:.4f}")

    # final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    inf_start = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    inf_time = (time.time() - inf_start) / len(all_labels)

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    return y_pred, y_true, inf_time, train_time, best_acc


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("Keyword Transformer (KWT-2) on MSWC Microset")
    print("Berg et al., Interspeech 2021")
    print("31-class keyword spotting with SpecAugment")
    print("=" * 70)

    if not os.path.isdir(CLIPS_DIR):
        print(f"ERROR: clips directory not found at {CLIPS_DIR}")
        return

    # load splits
    print(f"\n[1/4] Loading TRAIN split...")
    train_links, train_labels, kw_to_idx = load_split_csv(
        TRAIN_CSV, TARGET_KEYWORDS,
        CONFIG['samples_per_keyword_train'], seed=42
    )

    print(f"\n[2/4] Loading TEST split...")
    test_links, test_labels, _ = load_split_csv(
        TEST_CSV, TARGET_KEYWORDS,
        CONFIG['samples_per_keyword_test'], seed=42
    )

    print(f"\nDataset summary:")
    print(f"  Training files:  {len(train_links)}")
    print(f"  Test files:      {len(test_links)}")
    print(f"  Classes:         {len(TARGET_KEYWORDS)}")

    # build datasets with augmentation
    print(f"\n[3/4] Building datasets...")
    print("Training set (with augmentation):")
    train_dataset = MswcDataset(train_links, train_labels, CLIPS_DIR, augment=True)
    print("Test set (no augmentation):")
    test_dataset = MswcDataset(test_links, test_labels, CLIPS_DIR, augment=False)

    # train
    print(f"\n[4/4] Training KWT-2...")
    print("-" * 70)

    y_pred, y_true, inf_time, train_time, best_acc = train_kwt2(
        train_dataset, test_dataset, len(TARGET_KEYWORDS), CONFIG
    )

    # results
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "=" * 80)
    print("RESULTS: KWT-2 vs CNN BASELINE")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Params':<12}")
    print("-" * 85)
    print(f"{'CNN (baseline)':<25} {'86.7%':<12} {'87.6%':<12} {'86.7%':<12} {'86.9%':<12} {'~150K':<12}")
    print(f"{'KWT-2 (transformer)':<25} {f'{acc*100:.1f}%':<12} {f'{prec*100:.1f}%':<12} {f'{rec*100:.1f}%':<12} {f'{f1*100:.1f}%':<12} {'~2.4M':<12}")
    print(f"\nDelta: {(acc - 0.867)*100:+.1f}% {'(BEATS CNN)' if acc > 0.867 else '(below CNN)'}")
    print(f"MSWC paper DSCNN baseline: 85.3%  |  5-shot transfer: 94.7%")
    if acc > 0.947:
        print(">>> BEATS THE MSWC PAPER 5-SHOT BASELINE! <<<")
    elif acc > 0.853:
        print(f">>> Beats MSWC supervised baseline, {(0.947 - acc)*100:.1f}% below 5-shot <<<")

    # per-class report
    idx_to_kw = {v: k for k, v in kw_to_idx.items()}
    unique_labels = sorted(set(y_true))
    target_names = [idx_to_kw[i] for i in unique_labels]

    print(f"\n--- Classification Report (KWT-2) ---")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # timing
    print(f"\nTraining time: {train_time:.1f}s")
    print(f"Inference time per sample: {inf_time*1000:.4f}ms")

    # full comparison table
    print("\n" + "=" * 80)
    print("FULL MODEL COMPARISON (31-class MSWC microset)")
    print("=" * 80)
    models = [
        ('KWT-2 (transformer)', acc, '~2.4M'),
        ('CNN', 0.867, '~150K'),
        ('SVM (RBF)', 0.618, 'N/A'),
        ('Random Forest', 0.483, 'N/A'),
        ('KNN (k=5)', 0.370, 'N/A'),
    ]
    print(f"{'Model':<25} {'Accuracy':<12} {'Parameters':<12}")
    print("-" * 50)
    for name, a, params in sorted(models, key=lambda x: -x[1]):
        print(f"{name:<25} {a*100:.1f}%{'':<7} {params:<12}")

    # save results
    results = {
        'kwt2': {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'best_val_acc': float(best_acc),
            'inference_ms': float(inf_time * 1000),
            'train_time_s': float(train_time),
            'parameters': '~2.4M',
        },
        'baselines': {
            'cnn': 0.867,
            'svm': 0.618,
            'rf': 0.483,
            'knn': 0.370,
        },
        '_meta': {
            'architecture': 'KWT-2 (Berg et al., Interspeech 2021)',
            'dataset': 'MSWC English Microset',
            'n_classes': len(TARGET_KEYWORDS),
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'augmentation': 'SpecAugment + speed_perturb + time_shift',
            'optimizer': f"AdamW lr={CONFIG['lr']} wd={CONFIG['weight_decay']}",
            'schedule': f"cosine with {CONFIG['warmup_epochs']} epoch warmup",
            'epochs': CONFIG['epochs'],
            'batch_size': CONFIG['batch_size'],
        }
    }

    out_path = "kwt2_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
