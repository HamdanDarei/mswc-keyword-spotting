"""
Lab 9 Required Baselines: Random Forest and XGBoost
Compared against Transformer (KWT-2) on MSWC 31-class microset
Usage: pip install xgboost scikit-learn librosa numpy
       python xgboost_rf_baselines.py --data_path /path/to/mswc_microset/en
"""
import os, csv, time, argparse, numpy as np, librosa
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

KEYWORDS = ['backward','bed','bird','cat','dog','down','eight','five',
    'follow','forward','four','happy','house','learn','left','marvin',
    'nine','off','one','right','seven','sheila','six','stop','three',
    'tree','two','visual','wow','yes','zero']

def load_data(csv_path, clips_dir, max_per_kw=700):
    kw_to_idx = {kw: i for i, kw in enumerate(KEYWORDS)}
    by_kw = defaultdict(list)
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            if row['WORD'] in kw_to_idx: by_kw[row['WORD']].append(row['LINK'])
    rng = np.random.default_rng(42)
    X, y = [], []
    for kw in KEYWORDS:
        links = by_kw.get(kw, [])
        if len(links) > max_per_kw:
            links = rng.choice(links, max_per_kw, replace=False).tolist()
        for link in links:
            try:
                audio, _ = librosa.load(os.path.join(clips_dir, link), sr=16000, mono=True)
                if len(audio) == 0: continue
                mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=512, hop_length=160)
                if mfcc.shape[1] < 100: mfcc = np.pad(mfcc, ((0,0),(0,100-mfcc.shape[1])))
                else: mfcc = mfcc[:, :100]
                X.append(mfcc.flatten()); y.append(kw_to_idx[kw])
            except: continue
    return np.array(X), np.array(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.path.expanduser("~/Downloads/mswc_microset/en"))
    args = parser.parse_args()
    clips = os.path.join(args.data_path, "clips")
    train_csv = os.path.join(args.data_path, "en_train.csv")
    test_csv = os.path.join(args.data_path, "en_test.csv")

    print("Loading training data..."); X_train, y_train = load_data(train_csv, clips, 700)
    print(f"  Train: {X_train.shape}")
    print("Loading test data..."); X_test, y_test = load_data(test_csv, clips, 200)
    print(f"  Test: {X_test.shape}")

    print("\n--- Random Forest ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train); rf_time = time.time() - t0
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_f1 = f1_score(y_test, rf.predict(X_test), average='weighted')
    print(f"  Accuracy: {rf_acc*100:.1f}% | F1: {rf_f1:.3f} | Time: {rf_time:.1f}s")

    if HAS_XGB:
        print("\n--- XGBoost ---")
        t0 = time.time()
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
            objective='multi:softmax', num_class=31, eval_metric='mlogloss', random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train); xgb_time = time.time() - t0
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        xgb_f1 = f1_score(y_test, xgb_model.predict(X_test), average='weighted')
        print(f"  Accuracy: {xgb_acc*100:.1f}% | F1: {xgb_f1:.3f} | Time: {xgb_time:.1f}s")
