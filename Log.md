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

---
## Notes
