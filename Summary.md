# Project Summary — Korean Sign Language Recognition & Translation

## Overview

This project focused on the design, implementation, and evaluation of a research-grade pipeline for Korean Sign Language (KSL) recognition and translation.

Work progressed from isolated sign recognition toward continuous sign language recognition and sentence-level translation, with emphasis on building a functional, extensible system rather than producing a single model result.

The final system supports preprocessing, training, evaluation, and experimentation across multiple task formulations.

---

## Key Contributions

### 1. Dataset preprocessing and caching pipeline
- Designed and implemented offline preprocessing workflows for large-scale sign language datasets.
- Introduced caching mechanisms to avoid repeated video decoding and preprocessing.
- Supported multiple dataset formats with dataset-specific preprocessing modes.

Datasets supported:
- **NIASL2021** — isolated sign recognition (ISLR)
- **NIASLG1** — continuous sign language recognition (CSLR)

---

### 2. Isolated Sign Language Recognition (ISLR)
- Implemented ISLR as a classification task using a CNN–temporal encoder architecture.
- Integrated preprocessing, dataset loading, training, and evaluation into a single pipeline.
- Validated system correctness through large-scale experiments.

Key result:
- Successfully trained ISLR models on up to **100 word classes**, confirming pipeline stability and scalability.

---

### 3. Continuous Sign Language Recognition (CSLR)
- Extended the pipeline to support sentence-level sign language recognition.
- Integrated gloss-sequence modelling using **CTC-based decoding**.
- Resolved multiple runtime, environment, and initialisation issues through iterative debugging.
- Achieved the first fully working end-to-end CSLR execution.

Key result:
- Established a functional CSLR baseline with **~28% Word Error Rate (WER)**.

---

### 4. Sentence–gloss dataset construction
- Implemented tooling to extract gloss–sentence pairs from annotation spreadsheets.
- Normalised gloss segments into clean gloss sequences.
- Constructed reusable sentence–gloss datasets for downstream translation research.

---

### 5. Translation feasibility analysis
- Evaluated gloss-to-sentence translation using multiple large language models.
- Benchmarked translation quality using semantic similarity metrics:
  - BERTScore
  - COMET
- Used empirical results to inform future modelling direction.

---

### 6. System architecture and design decisions
- Identified reusable encoder components across ISLR and CSLR tasks.
- Decoupled encoding from decoding to support task-specific formulations.
- Documented limitations of CTC decoding for isolated recognition.
- Structured the codebase to support future end-to-end translation research.

---

## Final System Capabilities

The completed system supports:

- Offline preprocessing and caching of sign language video data
- ISLR training and evaluation (classification-based)
- CSLR training and evaluation (CTC-based sequence modelling)
- Sentence–gloss dataset generation
- Translation benchmarking using semantic metrics
- Scalable experimentation within a WSL-based workflow

---

## Outcome

By the end of the internship, the project delivered:

- A fully executable research pipeline
- Validated ISLR and CSLR task implementations
- Documented architectural decisions and limitations
- A strong foundation for future work on end-to-end sign language translation

This work prioritised system correctness, reproducibility, and clarity of design over isolated performance optimisation, enabling continued research and development beyond the internship period.

Read `Log.md` for more details.