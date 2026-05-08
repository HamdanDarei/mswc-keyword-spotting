"""
Keyword Spotting on MSWC Dataset
NeurIPS 2021 Datasets and Benchmarks Track

Compares CNN vs Classical ML (SVM, Random Forest, KNN)

Dataset: MLCommons/ml_spoken_words (loaded via Hugging Face)
Paper: Mazumder et al., "Multilingual Spoken Words Corpus", NeurIPS 2021
URL: https://mlcommons.org/datasets/multilingual-spoken-words/

Requirements:
    pip install datasets librosa torch scikit-learn numpy soundfile
"""

import numpy as np
import librosa
import time
import warnings
import os
warnings.filterwarnings('ignore')

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CNN will be skipped.")

# config
CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400,
    'n_mels': 40,
    'max_len': 100,
    'test_size': 0.2,
    'random_state': 42,
    'keywords': ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
    'samples_per_keyword': 200,
}


if TORCH_AVAILABLE:
    class KeywordCNN(nn.Module):
        """
        2D CNN for keyword spotting on MFCC spectrograms.

        Architecture:
            3 conv blocks (32 -> 64 -> 128 filters)
            each block: Conv2d(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
            followed by: Dense(128) -> ReLU -> Dropout(0.4) -> Dense(n_classes)
        """
        def __init__(self, n_mfcc=13, max_len=100, n_classes=10):
            super(KeywordCNN, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            # compute flattened size after 3 rounds of maxpool(2)
            h = max(1, n_mfcc // 8)
            w = max(1, max_len // 8)
            self.flat_size = 128 * h * w

            self.fc = nn.Sequential(
                nn.Linear(self.flat_size, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, n_classes)
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


def load_mswc_data(keywords, samples_per_keyword):
    """
    Load real MSWC data from Hugging Face.
    Downloads English wav subset and filters for target keywords.
    """
    print("Loading MSWC English dataset from Hugging Face...")
    print("(This may take a few minutes on first run as data is downloaded)")

    # load the english wav split
    # streaming=True avoids downloading the entire dataset at once
    ds = load_dataset(
        "MLCommons/ml_spoken_words",
        "en_wav",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # collect samples for each keyword
    keyword_set = set(keywords)
    kw_to_idx = {kw: i for i, kw in enumerate(keywords)}
    collected = {kw: 0 for kw in keywords}
    target_total = len(keywords) * samples_per_keyword

    X_audio = []
    y = []

    print(f"Filtering for {len(keywords)} keywords, {samples_per_keyword} samples each...")

    for sample in ds:
        word = sample["keyword"]
        if word not in keyword_set:
            continue
        if collected[word] >= samples_per_keyword:
            continue

        # extract audio array from the sample
        audio_data = sample["audio"]
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]

        # resample to 16kHz if needed
        if sr != CONFIG['sample_rate']:
            audio_array = librosa.resample(
                audio_array, orig_sr=sr, target_sr=CONFIG['sample_rate']
            )

        X_audio.append(audio_array)
        y.append(kw_to_idx[word])
        collected[word] += 1

        current = sum(collected.values())
        if current % 200 == 0:
            print(f"  Collected {current}/{target_total} samples...")

        # check if we have enough for all keywords
        if all(c >= samples_per_keyword for c in collected.values()):
            break

    # report what we got
    print(f"\nSamples collected per keyword:")
    for kw in keywords:
        print(f"  {kw}: {collected[kw]}")

    total = sum(collected.values())
    print(f"Total: {total} samples")

    if total < target_total:
        print(f"WARNING: Could only collect {total}/{target_total} samples.")
        print("Some keywords may have fewer samples than requested.")
        print("The model will still train on available data.")

    return X_audio, np.array(y), kw_to_idx


def extract_mfcc(audio, pad_to=100):
    """Extract MFCC features from audio waveform"""
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

    # pad or truncate to fixed length
    if mfcc.shape[1] < pad_to:
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_to - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :pad_to]

    return mfcc


def extract_stats(mfcc):
    """Extract statistical features from MFCC for classical ML models"""
    feats = []

    # per-coefficient stats
    for i in range(mfcc.shape[0]):
        feats.extend([
            np.mean(mfcc[i]),
            np.std(mfcc[i]),
            np.min(mfcc[i]),
            np.max(mfcc[i]),
            np.median(mfcc[i]),
        ])

    # delta features
    delta = librosa.feature.delta(mfcc)
    for i in range(delta.shape[0]):
        feats.extend([np.mean(delta[i]), np.std(delta[i])])

    # delta-delta features
    delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(delta2.shape[0]):
        feats.extend([np.mean(delta2[i]), np.std(delta2[i])])

    return np.array(feats)


def train_cnn(X_train, y_train, X_test, y_test, n_classes=10, epochs=40):
    """Train and evaluate CNN model"""
    if not TORCH_AVAILABLE:
        return None, None, None

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Training CNN on {device}...")

    X_tr = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    y_te = torch.LongTensor(y_test).to(device)

    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = KeywordCNN(CONFIG['n_mfcc'], CONFIG['max_len'], n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # track training history
    train_losses = []
    val_accs = []

    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_te).argmax(dim=1)
                acc = (preds == y_te).float().mean().item()
            val_accs.append(acc)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_acc={acc:.4f}")

    train_time = time.time() - train_start
    print(f"  Total training time: {train_time:.1f}s")

    # inference
    model.eval()
    with torch.no_grad():
        start = time.time()
        y_pred = model(X_te).argmax(dim=1).cpu().numpy()
        inf_time = (time.time() - start) / len(X_test)

    return y_pred, inf_time, train_time


def main():
    print("=" * 60)
    print("Keyword Spotting on MSWC (NeurIPS 2021)")
    print("CNN vs Classical ML Comparison")
    print("=" * 60)

    # load real MSWC data
    X_audio, y, kw_to_idx = load_mswc_data(
        CONFIG['keywords'], CONFIG['samples_per_keyword']
    )

    # extract features
    print("\nExtracting MFCC features...")
    X_mfcc = np.array([extract_mfcc(a, CONFIG['max_len']) for a in X_audio])
    X_stats = np.array([extract_stats(m) for m in X_mfcc])

    print(f"MFCC shape: {X_mfcc.shape}")
    print(f"Stats feature shape: {X_stats.shape}")

    # train/test split
    X_mfcc_train, X_mfcc_test, y_train, y_test = train_test_split(
        X_mfcc, y, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'], stratify=y
    )
    X_stats_train, X_stats_test, _, _ = train_test_split(
        X_stats, y, test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'], stratify=y
    )

    # scale features for classical ML
    scaler = StandardScaler()
    X_stats_train_scaled = scaler.fit_transform(X_stats_train)
    X_stats_test_scaled = scaler.transform(X_stats_test)

    print(f"\nTrain: {len(y_train)}, Test: {len(y_test)}")

    results = {}

    # train CNN
    print("\n" + "-" * 40)
    if TORCH_AVAILABLE:
        y_pred_cnn, inf_cnn, train_time_cnn = train_cnn(
            X_mfcc_train, y_train, X_mfcc_test, y_test,
            len(CONFIG['keywords'])
        )
        if y_pred_cnn is not None:
            results['CNN'] = {
                'accuracy': accuracy_score(y_test, y_pred_cnn),
                'precision': precision_score(y_test, y_pred_cnn, average='weighted'),
                'recall': recall_score(y_test, y_pred_cnn, average='weighted'),
                'f1': f1_score(y_test, y_pred_cnn, average='weighted'),
                'inference_ms': inf_cnn * 1000,
                'train_time_s': train_time_cnn,
                'y_pred': y_pred_cnn,
            }

    # train classical ML baselines
    classifiers = [
        ('SVM (RBF)', SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42)),
        ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5, weights='distance')),
    ]

    for name, clf in classifiers:
        print(f"\nTraining {name}...")
        t0 = time.time()
        clf.fit(X_stats_train_scaled, y_train)
        train_time = time.time() - t0

        start = time.time()
        y_pred = clf.predict(X_stats_test_scaled)
        inf_time = (time.time() - start) / len(y_test)

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'inference_ms': inf_time * 1000,
            'train_time_s': train_time,
            'y_pred': y_pred,
        }
        print(f"  {name}: acc={results[name]['accuracy']:.4f}, train={train_time:.1f}s")

    # results summary
    print("\n" + "=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    header = f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Inf(ms)':<10}"
    print(header)
    print("-" * 78)

    for name, r in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(
            f"{name:<20} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
            f"{r['recall']:<12.4f} {r['f1']:<12.4f} {r['inference_ms']:<10.4f}"
        )

    # per-class report for best model
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_pred = results[best_name]['y_pred']
    idx_to_kw = {v: k for k, v in kw_to_idx.items()}
    target_names = [idx_to_kw[i] for i in range(len(CONFIG['keywords']))]

    print(f"\n--- Classification Report ({best_name}) ---")
    print(classification_report(y_test, best_pred, target_names=target_names))

    print(f"\n--- Confusion Matrix ({best_name}) ---")
    cm = confusion_matrix(y_test, best_pred)
    # print header
    print(f"{'':>10}", end="")
    for kw in target_names:
        print(f"{kw:>7}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>10}", end="")
        for val in row:
            print(f"{val:>7}", end="")
        print()

    # training time comparison
    print(f"\n--- Training Time Comparison ---")
    for name, r in results.items():
        print(f"  {name}: {r['train_time_s']:.1f}s")

    # comparison with MSWC paper
    print("\n" + "=" * 78)
    print("COMPARISON WITH MSWC PAPER BASELINES")
    print("=" * 78)
    print("Original MSWC paper (Mazumder et al., NeurIPS 2021):")
    print("  - DSCNN on 7-class English: 85.3% accuracy")
    print("  - 5-shot transfer learning: 94.7% accuracy")
    print()
    print("Our results on 10-class task:")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  - {name}: {r['accuracy']*100:.1f}%")

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
