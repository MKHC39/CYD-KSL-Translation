## Data Preparation

1. from /preprocessing/ run KSL_dataset_cache.py (-h for args). (Currently only compatible with AIHUB NIASL2021)
2. Copy /preprocess/ directory into root


## 100 Word Trained Model

| Backbone | Dev Top-1 Acc  | Pretrained model                                             |
| -------- | ---------- | ----------- |
| ResNet18 | 93.49% | [[Google Drive]](https://drive.google.com/file/d/1fROBkaHk_u3cRPfnz-oZwGLxFpuciLsT/view?usp=sharing) |


## Gloss Sequence - Korean Benchmark Results

| Model           | BERTScore | COMET    |
|----------------|-----------|----------|
| GPT-5.2        | 0.92203   | 0.865952 |
| GPT-OSS        | 0.905586  | 0.808032 |
| Gemini 3 Flash | 0.928626  | 0.878687 |
| Gemini 3 Pro   | 0.930102  | 0.883101 |

**Dataset:** [glossTL/data.csv](glossTL/data.csv)
