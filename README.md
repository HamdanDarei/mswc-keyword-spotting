# Scaling Keyword Spotting Transformers to Thousand-Class Vocabularies on MSWC

**CSCI 4364 Machine Learning — Final Project**

**Team:** Hamdan Al Darei, Meshari Alanazi

## Project Description

This project evaluates how transformer-based keyword spotting scales from 31 to 6,210 classes on the Multilingual Spoken Words Corpus (MSWC). We implement the Keyword Transformer (KWT-2) from scratch, identify seven compounding errors that explain a 20-point accuracy gap versus published benchmarks, apply architectural corrections that improve accuracy from 77.6% to 81.2%, and fine-tune the Audio Spectrogram Transformer (AST) to investigate whether AudioSet pretraining transfers to keyword spotting.

**Key finding:** A from-scratch KWT-2 (2.9M params, 81.2%) outperforms a pretrained AST (87M params, 77.8%) on 1,000-class MSWC, demonstrating that AudioSet pretraining does not transfer well to short isolated speech.

## Setup and Installation

```bash
pip install torch torchaudio librosa scikit-learn numpy soundfile xgboost transformers accelerate
```

**Dataset:** Download MSWC English from [MLCommons](https://mlcommons.org/datasets/multilingual-spoken-words/)

```bash
# Full dataset (~124 GB compressed, ~33 GB English audio)
wget -qO- "https://mswc.mlcommons-storage.org/mswc.tar.gz" | tar -xz audio/en.tar.gz splits/en.tar.gz
mkdir -p mswc/audio mswc/splits
tar -xzf audio/en.tar.gz -C mswc/audio/
tar -xzf splits/en.tar.gz -C mswc/splits/
rm -f audio/en.tar.gz splits/en.tar.gz
```

Set the data path via command-line arguments (no hardcoded paths).

## Project Structure

```
code/
  keyword_spotting_kwt2.py          # KWT-2 on 31-class microset
  keyword_spotting_cnn_baselines.py # CNN + SVM + KNN baselines (31 classes)
  kwt2_scaling_study.py             # KWT-2 v1 scaling: 1,000 and 6,210 classes
  kwt2_v2.py                       # KWT-2 v2 corrected architecture
  xgboost_rf_baselines.py          # Random Forest + XGBoost comparison
  ast_finetune_mswc.py             # AST fine-tuning (87M params)
```

## How to Run

### Microset (31 classes)
```bash
python code/keyword_spotting_kwt2.py --data_path /path/to/mswc_microset/en
python code/xgboost_rf_baselines.py --data_path /path/to/mswc_microset/en
python code/keyword_spotting_cnn_baselines.py --data_path /path/to/mswc_microset/en
```

### Scaling Study (1,000 and 6,210 classes, requires GPU)
```bash
python code/kwt2_v2.py --data_path /path/to/mswc --precompute
python code/kwt2_v2.py --data_path /path/to/mswc --train
```

### AST Fine-Tuning (requires GPU with 12+ GB VRAM)
```bash
python code/ast_finetune_mswc.py --data_path /path/to/mswc
```

## Results

| Model | Classes | Accuracy | Params |
|---|---|---|---|
| Random Forest | 31 | 48.3% | N/A |
| XGBoost | 31 | 63.2% | N/A |
| CNN (3-block) | 31 | 86.7% | ~150K |
| KWT-2 v1 (13-bin MFCC) | 31 | 86.8% | 1.6M |
| KWT-2 v1 | 1,000 | 77.6% | 1.7M |
| KWT-2 v1 | 6,210 | 77.6% | 2.4M |
| **KWT-2 v2 (corrected)** | **1,000** | **81.2%** | **2.9M** |
| KWT-2 v2 | 6,210 | 77.0% | 5.6M |
| AST (freeze=8, AudioSet norm) | 1,000 | 63.4% | 87M |
| AST (freeze=4, MSWC norm) | 1,000 | 77.8% | 87M |

## Hardware Used

- MacBook Air M1 (CPU): microset experiments
- Desktop RTX 3060 12GB: full scaling study
- Google Colab A100 40GB: AST fine-tuning

## Relation to Paper Experiments

Each code file maps directly to a section of the paper:
- **Table 2 (Microset):** `keyword_spotting_kwt2.py` + `keyword_spotting_cnn_baselines.py` + `xgboost_rf_baselines.py`
- **Table 3 (v1 Scaling):** `kwt2_scaling_study.py`
- **Table 4 (v2 Corrections):** `kwt2_v2.py`
- **Section 4.5 (AST):** `ast_finetune_mswc.py`
- **Table 5 (Complete Results):** All scripts combined

## References

- Mazumder et al. (NeurIPS 2021) — Multilingual Spoken Words Corpus
- Berg et al. (Interspeech 2021) — Keyword Transformer
- Gong et al. (Interspeech 2021) — Audio Spectrogram Transformer
