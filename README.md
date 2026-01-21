# This is a fork of [CorrNet+](https://github.com/hulianyuyy/CorrNet_Plus), modified for Korean Sign Language as part of CYDInfoTech Internship project.

## Information
This repository contains all files relevent to CYDInfoTech's Internship project, commencement date 2025-12-15.
For detailed weekly log of the project, refer to [Log.md](Log.md#weekly-engineering-log)


## Data Preparation
Currently only [**NIASL2021**](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&searchKeyword=%EC%88%98%EC%96%B4%EB%B2%88%EC%97%AD&aihubDataSe=data&dataSetSn=103) and [**NIASLG1**](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&searchKeyword=%EC%88%98%EC%96%B4%EB%B2%88%EC%97%AD&aihubDataSe=data&dataSetSn=636) datasets are supported.
For more information on different datasets and research, refer to [**CYDProject**](CYDProject/)

1. from /preprocessing/ run KSL_dataset_cache.py (-h for args). (Currently compatible with AIHUB NIASL2021 and NIASLG1)
2. Copy /preprocess/ directory into project root.

## Training Model
Change /configs/KSL.yaml task value to "islr" for ISLR dataset, anything else for CLSR.


## 100 Word ISLR Trained Model

| Backbone | Dev Top-1 Acc  | Pretrained model                                             |
| -------- | ---------- | ----------- |
| ResNet18 | 93.49% | [[Google Drive]](https://drive.google.com/file/d/1fROBkaHk_u3cRPfnz-oZwGLxFpuciLsT/view?usp=sharing) |

## NIASLG1 COLDWAVE Trained Model

| Backbone | Dev WER | Pretrained model                                             |
| -------- |---------| ----------- |
| ResNet18 | 28.86%  | [[Google Drive]](https://drive.google.com/file/d/1tanTHvmSPkiWS-anJ_HvI7riRE26pb9O/view?usp=sharing) |



## Gloss Sequence - Korean Benchmark Results

| Model           | BERTScore | COMET    |
|----------------|-----------|----------|
| GPT-5.2        | 0.92203   | 0.865952 |
| GPT-OSS        | 0.905586  | 0.808032 |
| Gemini 3 Flash | 0.928626  | 0.878687 |
| Gemini 3 Pro   | 0.930102  | 0.883101 |

**Test sequence/results:** [glossTL/data.csv](glossTL/data.csv)
