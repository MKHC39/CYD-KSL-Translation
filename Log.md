# Internship Technical Log & Handover

## Table of Contents
- [Overview](#overview)
- [Project Scope](#project-scope)
- [Repository Structure](#repository-structure)
- [Weekly Engineering Log](#weekly-engineering-log)
- [Notes](#notes)

---

## Overview

This repository contains the research, implementation, and experimentation work completed during a six-week internship focused on Korean Sign Language (KSL) recognition and translation.

The project began with isolated sign language recognition (ISLR) and progressively expanded toward continuous sign language recognition (CSLR) and sentence-level translation. Work included:

- dataset preprocessing and caching
- model integration and refactoring
- evaluation pipeline design
- large-scale experimentation
- analysis of translation feasibility using large language models

The final system supports:

- **Isolated Sign Language Recognition (ISLR)**  
  using the **NIASL2021 dataset**, implemented as a classification task with a CNN–temporal encoder architecture.

- **Continuous Sign Language Recognition (CSLR)**  
  using the **NIASLG1 dataset**, implemented with gloss-sequence modelling and a **CTC-based decoder**.

- **Sentence–gloss dataset construction**  
  enabling extraction and normalisation of gloss–sentence pairs for downstream translation and analysis.

The repository is structured to support both experimentation and handover, with clear separation between preprocessing, modelling, evaluation, and research documentation.


---

## Project Scope

The scope of this project was to design, implement, and validate a functional pipeline for Korean Sign Language recognition and to investigate feasible pathways toward sentence-level translation.

The work was intentionally scoped to balance research exploration with practical system implementation.

### In scope

- **Dataset preprocessing and caching**
  - Support for both isolated-word and continuous sign language datasets.
  - Efficient preprocessing pipelines to enable scalable experimentation.

- **Isolated Sign Language Recognition (ISLR)**
  - Implementation and validation using the NIASL2021 dataset.
  - Model formulation as a classification task using a CNN–temporal encoder.
  - Large-scale training and evaluation to confirm system correctness.

- **Continuous Sign Language Recognition (CSLR)**
  - Integration of NIASLG1 dataset.
  - Gloss-sequence modelling using a CTC-based decoding framework.
  - End-to-end execution from preprocessing through evaluation.

- **Evaluation pipeline development**
  - Support for ISLR accuracy-based evaluation.
  - Support for CSLR sequence-level evaluation using Word Error Rate (WER).

- **Sentence–gloss dataset construction**
  - Extraction and normalisation of gloss–sentence pairs from annotation files.
  - Dataset preparation suitable for translation-level experimentation.

- **Translation feasibility analysis**
  - Empirical evaluation of gloss-to-sentence translation using large language models.
  - Quantitative comparison using semantic evaluation metrics (BERTScore, COMET).

### Out of scope

- **Full end-to-end sign language translation training**
  - Transformer-based SLT models were not trained within this project due to dataset size and pretraining requirements.

- **Large-scale pretraining**
  - No pretraining of visual encoders or language models was performed.

- **Production deployment**
  - The system was developed for research and experimentation purposes only.

- **Dataset creation**
  - No new sign language datasets were collected; all work used existing datasets.

This scope ensured the project delivered a fully functional research pipeline while maintaining realistic boundaries for time and computational constraints.


---

## Repository Structure

This repository contains the full codebase, data interfaces, and experiment tooling for Korean Sign Language recognition and translation research. The layout matches the structure of the original CorrNet+ codebase, with additional directories for project-specific documentation and experiments.

### Main directories

- **CYDProject/** *(Git submodule)*  
  Research material, reference notes, and sample datasets. Imported as a submodule to avoid tracking large files directly in the main repo.

- **configs/**  
  YAML configuration files, configured for the custom KSL dataset and feeder.

- **dataset/**  
  Dataset loader for model training and evaluation pipelines. Only includes KSL-specific feeder that bridge custom preprocessed outputs with the training framework without inclusion of original CorrNet+ feeder.

- **evaluation/slr_eval/**  
  CorrNet+ Utilities and evaluation scripts needed for sequence-level recognition metrics (e.g., WER calculation).

- **glossTL/**  
  Translation-layer tooling, including scripts for parsing gloss–sentence pairs, dataset construction, and LLM benchmarking.

- **modules/**  
  Model component implementations such as encoders, backbone networks, temporal layers, and loss definitions, modified from CorrNet+.

- **preprocessing/**  
  Custom preprocessing and caching scripts for NIASL2021 and NIASLG1 dataset to extract and store features from raw video and annotation data.

- **utils/**  
  Shared utilities used across the training and evaluation pipelines, such as decoding helpers, logging utilities, and performance tools.

### Root directory files

- **.gitattributes / .gitignore / .gitmodules**  
  Git configuration and ignore rules, including submodule definitions and excluded directories.

- **Log.md**  
  Detailed engineering log recording daily and weekly work; the primary handover document.

- **README.md**  
  Setup and usage instructions, including data preparation steps and link summaries for benchmarks.

- **environment.yml / requirements.txt**  
  Dependency manifests for environment reproducibility.

- **main.py / seq_scripts.py / slr_network.py**  
  Primary execution scripts used to launch training and evaluation runs.

---

### Notes on structure

- Preprocessing and dataset loading are separated to support efficient reuse and to avoid repeated video decoding.
- The translation pipeline (glossTL) is deliberately isolated from the base recognition pipeline.
- Config files and utilities centralise parameterisation and avoid hard-coding values in scripts.
- Submodule usage ensures large research materials don’t degrade Git performance.

This layout supports both **reproducible research workflows** and **clean project handover**.


---

## Weekly Engineering Log

- [Week 1 (15–19 December 2025)](#week1)
- [Weeks 2–3 (22–30 December 2025)](#week23)
- [Week 4 (5-9 January 2026)](#week4)
- [Week 5 (13–16 January 2026)](#week5)
- [Week 6 (19–20 January 2026)](#week6)

<a id="week1"></a>
### Week 1 (15–19 December 2025)

#### Focus  
Preliminary research, dataset familiarisation, literature review, and foundational experimentation.

The objective of Week 1 was to build sufficient theoretical and practical grounding before committing to implementation work. This week deliberately prioritised understanding over development in order to reduce architectural uncertainty later in the project.

---

#### Summary of work

**Repository setup and planning (15 Dec)**  
- Created and configured the initial Git repository.  
- Established baseline documentation and note-taking structure.  
- Began consolidating research materials and project references.  
- Explored early task-tracking approaches and refined workflow preferences.

**Dataset inspection and structural understanding (16–17 Dec)**  
- Introduced sample data from two distinct Korean Sign Language datasets:
  - **NIASL2021** (stored under `New_sample`)
  - **NIASLG1** disaster dataset (stored under `재난 sample`)
- Added official dataset documentation for both datasets, one per dataset, to serve as authoritative structural references.
- Inspected and compared dataset characteristics, including:
  - directory hierarchies
  - file naming conventions
  - modality separation (video, keypoints, linguistic annotations, spreadsheets)
- Confirmed that the two datasets are structurally different and not directly interchangeable.
- Identified that future preprocessing and dataset loaders would need to be dataset-specific.

**Documentation refinement and literature consolidation (17–18 Dec)**  
- Reorganised documentation as understanding matured:
  - replaced general-purpose resource files with a focused `SLR Papers.md`
  - introduced prioritised reading lists
  - updated planning notes to reflect clearer dataset distinctions
- Conducted an initial literature review focused on sign language recognition (SLR).
- Expanded the **RGB SLR** section with papers that were read and analysed.
- Created and maintained a structured **Summary** table within `SLR Papers.md`, comparing papers across:
  - model architectures
  - benchmark datasets
  - evaluation metrics
  - availability of source code
  - system structure
  - additional implementation notes
- Shifted from paper-by-paper reading to comparative analysis to support future design decisions.

**Hands-on experimentation outside the repository (19–21 Dec)**  
- Conducted independent prototyping in a separate personal Python project to avoid polluting the main research repository.
- Implemented a small-scale **ASL alphabet recognition system** using **YOLOv11** and a public Roboflow dataset.
- Built custom PyTorch `Dataset` and `DataLoader` classes to explore:
  - data preprocessing workflows
  - batch construction
  - label representation
  - interaction between dataset structure and model training
- Used object detection as a controlled environment to develop intuition about:
  - convolutional feature learning
  - annotation-driven supervision
  - training dynamics and loss behaviour

---

#### Technical outcomes

By the end of Week 1:

- Dataset structures for both NIASL2021 and NIASLG1 were understood at a practical level.
- Official documentation was consolidated for dataset interpretation.
- A structured literature review framework was established to guide architectural decisions.
- Practical familiarity with PyTorch data pipelines and model training was developed through independent experimentation.
---

#### Outcome

Week 1 primarily focused on building foundational understanding rather than producing implementation artifacts.

By the end of the week, the following learning objectives had been achieved:

- Developed a clearer understanding of **how deep neural networks (DNNs) operate**, particularly in vision-based tasks.
- Gained familiarity with **how existing research papers implement sign language recognition (SLR) networks**, including common architectural patterns.
- Built awareness of **basic structural components and implementation methods** used across SLR systems.
- Formed an initial mental model of **how such systems could realistically be replicated**, rather than treated as black-box models.
- Learned and contextualised key technical terminology commonly used in the field, including:
  - glosses and gloss-level annotation
  - convolutional neural networks (CNNs)
  - BiLSTM-based temporal modelling
  - 1D, 2D, and 3D convolutional architectures and their respective use cases

This week established the conceptual vocabulary and structural intuition required to meaningfully engage with sign language recognition and translation research in subsequent implementation phases.

<a id="week23"></a>
### Weeks 2–3 (22–30 December 2025)

#### Focus  
Implementation of the preprocessing pipeline, dataset abstraction, and integration with an existing SLR framework (CorrNet+).

Following the preliminary research phase in Week 1, Weeks 2–3 marked the transition into active system construction. Work during this period focused on transforming conceptual understanding into executable code, with particular emphasis on data preprocessing, performance, and framework compatibility.

---

#### Summary of work

**Initial preprocessing and dataset scaffolding (22 Dec)**  
- Established the implementation-oriented repository structure.
- Introduced dedicated modules for:
  - preprocessing
  - dataset handling
  - model components
- Implemented early preprocessing scripts for:
  - video frame extraction
  - border handling and cropping
  - clip-level processing
- Created initial dataset abstractions compatible with PyTorch (`Dataset`, `DataLoader`).
- Introduced clip-level caching logic to separate expensive preprocessing from runtime data loading.

**Design-driven caching decision (22–23 Dec)**  
- Early testing revealed that performing preprocessing inside `__getitem__` was prohibitively slow, as each sample access required reopening and reprocessing full video files.
- To resolve this, preprocessing was decoupled from data loading:
  - preprocessing is executed once per clip
  - processed outputs are cached to disk
  - dataset loaders consume cached tensors directly
- This decision fundamentally shaped the dataset architecture used throughout the remainder of the project.

**Pipeline refinement and stabilisation (23 Dec)**  
- Refined preprocessing scripts to improve correctness and robustness.
- Iteratively updated dataset loaders to align with actual cached data formats.
- Scanned for unclean data in dataset that can result in errors.
- Resolved mismatches between preprocessing outputs and dataset assumptions discovered during runtime testing.

**Preparation for CorrNet+ injection (24 Dec)**  
- Began restructuring code to match CorrNet+ expectations around:
  - dataset interfaces
  - argument structure
  - module layout
- Introduced KSL-specific dataset modules to mirror CorrNet-style imports.
- Added early dictionary and label-mapping artefacts required for dataset indexing.
- Removed standalone entrypoints in favour of framework-style script execution.
- Transitioned the project from a standalone prototype toward an injectable architecture.

**CorrNet+ injection completion and evaluation-stage debugging (29 Dec)**  
- Completed CorrNet+ injection to the point where the framework could successfully:
  - load the custom cached KSL dataset
  - accept the custom `KSL_pydata.py` dataset feeder
  - execute the training/inference pipeline without crashing at data ingestion
- Confirmed that dataset preprocessing, caching, and loading were functionally compatible with CorrNet’s internal data flow.
- Encountered failures specifically during the **evaluation stage**, rather than during training or forward execution.
- Investigated and patched evaluation-time incompatibilities, including:
  - missing or empty arguments expected by CorrNet evaluation utilities
  - absence of STM files required by sequence-level evaluation code paths
- Implemented compatibility features such as:
  - placeholder / empty argument handling
  - STM generation utilities to satisfy CorrNet evaluation assumptions
- This clarified the boundary between successful framework injection and unresolved evaluation constraints tied to task formulation.

**Separate ISLR evaluation implementation and validation (30 Dec)**  
- Implemented a **separate ISLR evaluation function** independent of CorrNet’s default evaluation pipeline.
- This was motivated by persistent evaluation-stage failures whose root cause was not yet understood at the time.
- The new evaluation function was designed to:
  - directly consume outputs from the custom cached KSL dataset
  - bypass parts of CorrNet’s tightly coupled evaluation assumptions
- Successfully validated the evaluation function on a **small-scale test dataset**, confirming that:
  - decoding logic executed end-to-end
  - evaluation metrics could be computed without runtime failure
- Although the function did not yet scale cleanly to the full dataset, it provided a controlled environment for isolating evaluation-related issues.

This work established an alternative evaluation pathway, enabling continued progress despite unresolved compatibility issues in the original framework.

---

#### Technical outcomes

By the end of Weeks 2–3:

- A complete preprocessing pipeline was implemented for video-based sign language data.
- Dataset abstraction and caching were established as core architectural components, enabling repeated experimentation without reprocessing raw videos.
- The custom cached KSL dataset was successfully injected into the CorrNet+ framework, with the framework able to:
  - load the custom dataset feeder
  - execute training and inference without crashing at data ingestion
- Evaluation-stage failures were identified as the primary remaining blocker, occurring after successful model execution.
- Compatibility features were introduced to reduce evaluation-time failures, including:
  - placeholder handling for missing or empty arguments
  - STM generation to satisfy sequence-evaluation requirements
- A separate ISLR evaluation function was implemented to isolate evaluation behaviour from CorrNet’s internal pipeline.
- The custom evaluation function was able to run end-to-end on a small-scale test dataset (10 word classes, 5 angles, 1 signer for 50 samples)..

At this stage, the exact cause of evaluation inconsistencies was not yet understood.

---

#### Outcome

Weeks 2–3 established the project’s **first runnable and testable system state**.

By the end of this phase:
- preprocessing, caching, and dataset loading were functionally stable
- CorrNet+ could be executed using the custom KSL dataset
- evaluation failures were narrowed down to sequence-level evaluation logic rather than preprocessing or data ingestion

Although evaluation results were inconsistent, the underlying cause had not yet been identified at this point.

The focus in this phase was to establish a fully runnable system end-to-end baseline. While that goal had been achieved, it was clear that further work was required to diagnose and resolve evaluation-stage behaviour — which became the focus of the following week.

---

<a id="week4"></a>
### Week 4 (5–9 January 2026)

#### Focus  
Repository restructuring, large-scale validation of the ISLR pipeline, diagnosis of evaluation failures, and transition planning toward sequence-level translation.

Following the initial implementation phase in Weeks 2–3, Week 4 focused on consolidating the codebase into a full CorrNet-style structure, validating the system at scale, and reassessing architectural direction based on empirical results.

---

#### Summary of work

**Repository restructuring and scale-up (5 Jan)**  
- Restructured the project into a full CorrNet-style codebase with KSL-specific patches applied directly into the framework.
- Transitioned from patch-based injection toward treating KSL as a first-class dataset within the repository.
- Expanded experimental scale significantly:
  - from small sanity runs (10 word classes, ~50 samples, no dev set)
  - to a medium-scale dataset (20 word classes, 5 angles, 16 training signers + 2 dev signers, ~1,800 samples)
- This scale-up exposed evaluation behaviour that was not observable under smaller experimental conditions.

**Evaluation failure diagnosis (6 Jan)**  
- Conducted focused debugging on NaN and evaluation crashes observed during large-scale runs.
- Identified the root cause as CTC decoder collapse for sequences of temporal length 1:
  - logits collapsed into empty sequences
  - infinite losses were produced but not always surfaced
  - blank entries propagated into evaluation, causing index mismatches
- Determined that the issue affected both evaluation and training, invalidating the CTC-based ISLR formulation under current dataset constraints.

**Architectural reassessment and pivot (6 Jan)**  
- Reviewed state-of-the-art modelling approaches across:
  - ISLR
  - CSLR
  - end-to-end SLT
- Identified a consistent encoder–decoder pattern across tasks.
- Concluded that CorrNet’s encoder (2D CNN + 1D temporal convolution + BiLSTM) remained reusable and valuable.
- Decided to decouple decoding from the encoder:
  - retain the encoder
  - replace CTC-based decoding with task-specific heads
- Defined a revised ISLR formulation using cross-entropy loss rather than CTC.

**ISLR pipeline refactor (7 Jan)**  
- Introduced an explicit `mode=islr` execution path within the training and evaluation scripts.
- Refactored `seq_train` and `seq_eval` to support task-specific logic.
- Removed or bypassed CTC-dependent code paths.
- Modified module interfaces to return logits directly for simplified downstream handling.
- Initiated overnight training on the revised ISLR formulation.

**ISLR validation and project direction alignment (8 Jan)**  
- Successfully trained the 20-word ISLR dataset, achieving **98.5% top-1 accuracy**.
- Confirmed stability and correctness of the revised encoder + cross-entropy formulation.
- Consulted with the project manager to review progress and define next-stage direction.
- Agreed to shift focus toward sequence-level modelling as the final project goal.
- Defined a decision framework:
  - evaluate feasibility of gloss-sequence translation using LLMs
  - if insufficient, pursue end-to-end Transformer-based SLT
- Began preparing for sequence-level experimentation by:
  - downloading the NIASLG1 dataset
  - reintegrating early research material (`CYDProject`) into the main repository
  - launching larger ISLR runs (100-word dataset, ~9,000 samples) in parallel.

**Translation-layer preprocessing groundwork (9 Jan)**  
- Initiated development of translation-focused preprocessing tools under `glossTL/`.
- Implemented scripts to parse XLSX files containing gloss–sentence pairs.
- Added validation tooling to detect malformed or inconsistent annotations.
- Used Jupyter-based workflows to visualise and inspect sequence-level data.
- Cleaned generated preprocessing artefacts from version control and stabilised repository hygiene.

---

#### Technical outcomes

By the end of Week 4:

- The project was fully restructured into a CorrNet-style framework with KSL treated as a first-class dataset.
- A stable and scalable ISLR pipeline was established, validated at both medium and large scale.
- Fundamental limitations of CTC-based decoding for isolated sign recognition were identified and resolved through architectural redesign.
- The CorrNet encoder was confirmed as a reusable backbone across multiple task formulations.
- A clear transition path from recognition toward sequence-level translation was defined.
- Tooling for gloss–sentence dataset inspection and preprocessing was in place.

---

#### Outcome

Week 4 marked a decisive transition from framework integration to **architectural clarity**.

By the end of the week:
- isolated sign recognition was functionally solved within the project scope
- remaining challenges were no longer related to system stability, but to task formulation and modelling strategy
- the project direction shifted decisively toward sequence-level translation research

This phase established a robust foundation for subsequent work on CSLR and end-to-end sign language translation, grounded in validated components and informed design decisions rather than trial-and-error experimentation.

---
<a id="week5"></a>
### Week 5 (13–16 January 2026)

#### Focus  
Transition from isolated recognition toward sequence-level translation through dataset construction, benchmarking, and major workflow restructuring.

Week 5 marked a shift away from model-centric debugging and toward **data and workflow readiness**. The emphasis moved to constructing sentence–gloss datasets, evaluating translation feasibility using large language models, and restructuring the development environment to support scalable CSLR and SLT experimentation.

---

#### Summary of work

**Sentence–gloss extraction pipeline development (13–14 Jan)**  
- Started development of `glossTL/pull_script.py` to parse sentence–gloss annotations stored in XLSX files.
- Implemented a full extraction pipeline:
  - XLSX files loaded via `pandas.read_excel`
  - converted into DataFrame format
  - gloss-level segments (label, start time, end time) parsed and ordered
  - segments concatenated into full gloss sequences
- Constructed a normalised dataset containing:
  - gloss sequences
  - corresponding natural-language sentences
  - associated metadata
- Exported processed data into JSON format and materialised the dataset as `ksl_sentence_gloss.json`.
- Added validation and inspection tooling to detect malformed or inconsistent annotation entries.
- Used Jupyter-based workflows to visualise dataset structure and verify assumptions.

**Gloss-to-sentence translation experiments (14–15 Jan)**  
- Extracted gloss–sentence pairs from the constructed dataset.
- Evaluated gloss-to-sentence translation using multiple large language models.
- Logged predicted sentences alongside ground-truth references in spreadsheet form.
- Implemented semantic evaluation metrics:
  - BERTScore
  - COMET
- Completed benchmarking across four models, enabling comparative analysis of translation quality beyond surface-level string matching.

**Repository maintenance and stability improvements (15 Jan)**  
- Identified severe Git performance issues caused by tracking a nested research repository containing ~13,000 files.
- Removed the nested repository from direct tracking.
- Reintroduced it correctly as a Git submodule, restoring repository responsiveness and maintainability.

**Workflow refinement and dataset normalisation (15–16 Jan)**  
- Continued iterative refinement of extraction logic to streamline the end-to-end workflow.
- Improved handling of noisy annotations and inconsistent formatting.
- Implemented gloss sequence cleanup logic to merge overlapping or duplicated temporal segments
  (e.g. overlapping identical gloss labels merged into a single continuous segment).
- This step was necessary to generate clean ground-truth sequences suitable for CSLR training.

**Major environment and performance restructuring (16 Jan)**  
- Migrated the entire project from the Windows filesystem (`/mnt/c/...`) into the native WSL filesystem (`/home/...`).
- Relocated all datasets and cached preprocessing outputs into WSL.
- This eliminated cross-filesystem I/O overhead and resulted in significant performance improvements for:
  - preprocessing scripts
  - dataset loading
  - caching
  - long-running training jobs
- Prepared the environment for scalable sequence-level training.

---

#### Technical outcomes

By the end of Week 5:

- A complete sentence–gloss extraction pipeline was implemented and validated.
- Translation feasibility using large language models was empirically evaluated.
- Objective semantic evaluation metrics (BERTScore and COMET) were integrated.
- Gloss-sequence ground truth was cleaned and normalised for CSLR training.
- The entire development workflow was migrated into a high-performance WSL-native environment.
- Repository structure and Git performance were stabilised.

---

#### Outcome

Week 5 marked a decisive transition from isolated sign recognition toward **sequence-level modelling and translation**.

By the end of the week:
- the project had shifted focus from model debugging to data readiness
- sentence–gloss datasets suitable for CSLR and SLT experimentation were available
- translation strategies could be evaluated empirically rather than assumed
- the development environment was capable of supporting large-scale training workloads

This phase established the data, tooling, and workflow foundation required for subsequent work on continuous sign language recognition and end-to-end sign language translation.

---
<a id="week6"></a>
### Week 6 (19–20 January 2026)

#### Focus  
Integration and stabilisation of continuous sign language recognition (CSLR), transitioning from dataset preparation toward executable sequence-level training.

Week 6 focused on extending the existing pipeline to support CSLR datasets, resolving runtime and environment-level issues, and achieving the first successful end-to-end CSLR run.

Commencement of documentation and logging work started this week.

---

#### Summary of work

**NIASLG1 preprocessing integration (19 Jan)**  
- Finalised refactoring of preprocessing scripts to introduce a dedicated NIASLG1 dataset mode.
- Extended caching and preprocessing logic to support:
  - sentence-level video samples
  - gloss-sequence ground truth
- Enabled the preprocessing pipeline to handle continuous sign language data rather than isolated-word samples.

**CSLR framework preparation and initial training attempts (19 Jan)**  
- Added missing CorrNet CSLR components required for training and evaluation.
- Integrated CSLR-related modules into the existing execution pipeline.
- Performed multiple CSLR training attempts.
- Failures were primarily caused by:
  - small initialisation bugs only observable during runtime
  - WSL environment constraints (disk space and memory allocation)
- Preprocessing outputs and dataloader logic were not identified as the root cause.

**CSLR stabilisation and first successful run (20 Jan)**  
- Continued iterative bug-fixing across training scripts, configuration files, and execution flow.
- Resolved remaining runtime blockers preventing CSLR from completing.
- Achieved the first successful end-to-end CSLR training and evaluation run using a small sample NIASLG1 dataset.
- Recorded initial performance:
  - Word Error Rate (WER): ~28%
- Confirmed functional integration of:
  - preprocessing
  - dataset loading
  - model execution
  - decoding
  - evaluation.

**Documentation initiation (20 Jan)**  
- Began writing internal documentation (`Log.md`) to record:
  - workflow decisions
  - known issues
  - implementation context
- Established a basis for structured handover documentation.

---

#### Technical outcomes

By the end of 20 January:

- Preprocessing and caching supported both ISLR and CSLR datasets.
- The CSLR pipeline was fully executable end-to-end.
- Runtime and environment-level blockers were identified and resolved.
- A first quantitative CSLR baseline (28% WER) was established.
- Documentation efforts were initiated to support continuity and handover.

---

#### Outcome

Week 6 marked the project’s transition from **dataset readiness to functional sequence-level recognition**, with emphasis of preparing for project handover.

By the end of this period:
- CSLR training was no longer theoretical and could be executed in practice
- the full pipeline from preprocessing through evaluation was operational
- documentation work commenced for project handover

This established a proof-of-concept baseline demonstrating that end-to-end translation from sign language sequences to Korean sentences is theoretically feasible, with a fully functional system skeleton in place.

---

## Notes

This section documents important implementation details, limitations, and assumptions that are not immediately obvious from code or commit history. These notes are intended to support future development and prevent common sources of confusion.

---

### Dataset-specific assumptions

- **NIASL2021 (ISLR)** and **NIASLG1 (CSLR)** differ significantly in structure and cannot share preprocessing logic.
- Preprocessing scripts are dataset-specific by design and should not be unified without careful inspection of annotation formats.
- File naming conventions, annotation formats, and metadata structures differ across datasets.
- The argument for dataset is passed via command-line arguments at runtime and is required.

---

### Preprocessing and caching design

- All video preprocessing is performed **offline** and cached to disk.
- Runtime preprocessing inside `Dataset.__getitem__` was intentionally avoided due to severe performance degradation.
- Any modification to preprocessing scripts requires regeneration of cached files.

---

### CTC-related limitations

- CTC decoding is highly sensitive to temporal sequence length.
- Sequences with effective temporal length of 1 can collapse into empty outputs during decoding.
- This behaviour previously caused:
  - silent NaN loss propagation
  - blank evaluation entries
  - index mismatches during evaluation
- These failures may not surface during small-scale tests and only appear under larger datasets.

---

### ISLR vs CSLR formulation

- ISLR is implemented as a **classification task** using cross-entropy loss.
- CSLR is implemented as a **sequence modelling task** using CTC decoding.
- Although both tasks share the same encoder backbone, their decoding and evaluation pipelines are intentionally separate.
- Attempting to unify ISLR and CSLR decoding logic is strongly discouraged.

---

### Encoder reuse assumption

- The CorrNet encoder (2D CNN + temporal convolution + BiLSTM) is treated as a reusable backbone.
- Decoder heads and evaluation logic are task-specific.
- This design allows:
  - ISLR
  - CSLR
  - potential end-to-end SLT
  to share feature extraction while diverging at decoding.

---

### Translation experiments

- Gloss-to-sentence translation experiments were conducted for feasibility analysis only.
- Large language models were evaluated using extracted gloss sequences, not raw video.
- Evaluation metrics (BERTScore, COMET) measure semantic similarity rather than grammatical correctness.
- These experiments do not constitute a production translation system.

---

### Environment considerations

- Running preprocessing or training from the Windows filesystem (`/mnt/c`) inside WSL causes severe I/O slowdown.
- The project, datasets, and caches should reside entirely within the native WSL filesystem (`/home/...`).
- Insufficient WSL disk or memory allocation will cause hard crashing of the whole WSL system.

---

### Git and repository management

- The `CYDProject` directory is maintained as a Git submodule to prevent repository performance degradation.
- Generated directories such as `cache/`, `work_dir/`, and preprocessing outputs are intentionally ignored.

---

### Known limitations

- No large-scale pretrained models were used.
- End-to-end sign language translation models were not trained due to dataset and computational constraints.
- Current CSLR performance (~28% WER) represents an early baseline rather than an optimised result.
- Current models for both ISLR and CSLR were only trained on a small sample of the dataset.

---

### Intended future direction

Potential next steps include:

- CSLR model optimisation and hyperparameter tuning
- investigation of transformer-based sequence modelling
- integration of gloss-level CSLR outputs with downstream translation models
- exploration of end-to-end sign language translation with pretrained encoders
- Training with larger scale datasets and larger models for accuracy

