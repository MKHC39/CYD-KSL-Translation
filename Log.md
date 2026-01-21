# Internship Technical Log & Handover

## Table of Contents
- [Overview](#overview)
- [Project Scope](#project-scope)
- [Repository Structure](#repository-structure)
- [Weekly Engineering Log](#weekly-engineering-log)
- [Notes](#notes)

---

## Overview

---

## Project Scope

---

## Repository Structure

---

## Weekly Engineering Log

- [Week 1 (15–19 December 2025)](#week1)
- [Weeks 2–3 (22–30 December 2025)](#week23)

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
- The custom evaluation function was able to run end-to-end on a small-scale test dataset.

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
## Notes
