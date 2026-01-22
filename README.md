# This is a fork of [CorrNet+](https://github.com/hulianyuyy/CorrNet_Plus), modified for Korean Sign Language as part of CYDInfoTech Internship project.

## Overview
This repository contains all files relevent to CYDInfoTech's Internship project, commencement date 2025-12-15.
For detailed weekly log of the project, refer to [Log.md#Weekly Engineering Log](Log.md#weekly-engineering-log)

This project focused on the design, implementation, and evaluation of a research-grade pipeline for Korean Sign Language (KSL) recognition and translation.

Work progressed from isolated sign recognition toward continuous sign language recognition and sentence-level translation, with emphasis on building a functional, extensible system rather than producing a single model result.

The final system supports preprocessing, training, evaluation, and experimentation across multiple task formulations.

For a summary of the full project, refer to [summary.md](summary.md)

For information on the project's structure, refer to [Log.md](Log.md#repository-structure)

## Outcome

By the end of the internship, the project delivered:

- A fully executable research pipeline
- Validated ISLR and CSLR task implementations
- Documented architectural decisions and limitations
- A strong foundation for future work on end-to-end sign language translation

This work prioritised system correctness, reproducibility, and clarity of design over isolated performance optimisation, enabling continued research and development beyond the internship period.

## Initial Research
An extensive review of existing research on different techniques of SLR (Sign Language Recognition) and SLT (Sign Language Translation) was conducted.
From this, an understanding of different state-of-the-art approaches was gained and different methods of implementation were researched, comparing strengths and weaknesses for each method.
Sign Language Translation can be largely divided into three different subsets:
- Isolated Sign Language Recognition (ISLR)
- Continuous Sign Language Recognition (CSLR)
- Language Translation (SLT)

Common in all three tasks is the need for preprocessing to convert the data into a format suitable for training. The full list of papers reviewed is available in [**SLR Papers.md**](CYDProject/SLR Papers.md) alongside a summary table including available online datasets and model evaluations.

## Data Preparation
Currently only [**NIASL2021**](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&searchKeyword=%EC%88%98%EC%96%B4%EB%B2%88%EC%97%AD&aihubDataSe=data&dataSetSn=103) and [**NIASLG1**](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&searchKeyword=%EC%88%98%EC%96%B4%EB%B2%88%EC%97%AD&aihubDataSe=data&dataSetSn=636) datasets are supported.
For more information on different datasets and research, refer to [**CYDProject**](CYDProject/)

This converts the mp4 files into a cache of stacked tensor ndarrys with label ids and generates gloss_dict.npy and other files required for training.

1. Extract the full dataset into dataset root directory (NIASL2021/NIASLG1).
   1. (for NIASLG1) run ```python -m glossTL.pull_script```, enter the path to the dataset directory containing all xlsx files.
2. run 
    ``` bash
   # run -h for full list of args
   python -m preprocessing.KSL_dataset_cache.py --dataset (NIASLG1,NIASL2021)
   ```
3. Copy `cache_root/preprocess/` directory into project root so that the project directory looks like `project/preprocess/KSL/...`.

Tool [`preprocessing/read_npy.py`](preprocessing/read_npy.py) is provided to check the data of the generated npy files.

## Training Model
1. Change [`/configs/KSL.yaml`](configs/KSL.yaml) task value to "islr" for ISLR dataset, anything else for CLSR.
2. run ```python main.py```

The models will be saved to `/work_dir/`

## Pretrained Models

### ISLR

#### Words 1-100 NIASL2021 Model

100/3,000 (3.3%) Word Classes used to train the model. 

| Backbone | Dev Top-1 Acc  | Pretrained model                                             |
| -------- | ---------- | ----------- |
| ResNet18 | 93.49% | [[Google Drive]](https://drive.google.com/file/d/1fROBkaHk_u3cRPfnz-oZwGLxFpuciLsT/view?usp=sharing) |

---
### CSLR

#### NIASLG1 COLDWAVE Words 1-400 Model

1,015/6,531 (15.5%) of COLDWAVE data used to train the model (from Word ID 1-400).

| Backbone | Dev WER | Pretrained model                                             |
| -------- |---------| ----------- |
| ResNet18 | 27.97%  | [[Google Drive]](https://drive.google.com/file/d/1-R-AeWW75z2ZIeIas1omnBwwH4pdFles/view?usp=sharing) |

---
## Gloss Sequence - Korean Benchmark Results
The feasibility of using the CSLR models and feeding the output to already available LLMs to generate full Korean sentences has been tested. 
To test this yourself, run 
```bash
python -m glossTL.pull_script
python -m glossTL.read_json
```

Copy output and paste into LLM model for testing.

### Benchmarking
To benchmark the performance of the models create a csv file that matches the format in [data.csv](glossTL/data.csv).
Edit [`back_to_json.py`](glossTL/back_to_json.py) to match the column of your csv file to generate the jsonl files of respective models.
Edit [`BERTSCORE.py`](glossTL/BERTSCORE.py) and [`COMET.py`](glossTL/COMET.py) to match the model names, then run.

### Results


| Model           | BERTScore | COMET    |
|----------------|-----------|----------|
| GPT-5.2        | 0.92203   | 0.865952 |
| GPT-OSS        | 0.905586  | 0.808032 |
| Gemini 3 Flash | 0.928626  | 0.878687 |
| Gemini 3 Pro   | 0.930102  | 0.883101 |

**Test sequence/results:** [glossTL/data.csv](glossTL/data.csv)
