# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0]

## [0.0.6] - 2026-1-07

### Added
- `dataset/KSL_pydata.py` custom DataLoader for NIASL2021 dataset compatible with CorrNet.

### Changed
- `main.py` to use `KSL_pydata.py` for KSL.
- `seq_scripts.py` check for islr mode.
- `seq_scripts.py` for new islr evaluation metric and scripts.

## [0.0.5] - 2025-12-30

### Added
- `preprocessing/preprocess_clip.py` to convert video clips to stacked tensor.
- `preprocessing/KSL_dataset_cache.py` to implement caching to speed up dataset loading.

## [0.0.4] - 2025-12-24

### Added
- `preprocessing/border.py` and `preprocessing/frame_extract.py` for extracting frames from video files.

## [0.0.3] - 2025-12-22

### Added
- Implemented data augmentation techniques for training. [999bedc](https://github.com/MKHC39/CYD-KSL-Translation/commit/999bedcadae3b6274df856d576d1412276ae399c)
- Submodule `CYDProject`: Augmentation techniques. [34edd70](https://github.com/MKHC39/CYDProject/commit/34edd70e2e937d0e24ead43b5144395e445536e6)
- Experimental data augmentation techniques implemented into ASL project.
- Started work on preprocessing scripts for KSL NIASL2021 ISLR dataset.


## [0.0.1] - 2025-12-21

### Added
- Started work on a small scale ASL alphabet recognition using YOLOv11
  - Compiled torch.Dataset and DataLoader classes for training and evaluation.
  - Object recognition framework implemented using YOLOv11.
  - Dataset supplied by [Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters/1)

## [0.0.0] - 2025-12-18

### Added
- Initial project resources and paper organization. [41fc621](https://github.com/MKHC39/CYD-KSL-Translation/commit/41fc6212bcdbb360d1f99aa21441af9ec56c9a60)
- Repository setup with initial project files. [a3b28c0](https://github.com/MKHC39/CYD-KSL-Translation/commit/a3b28c02206597f4d56f795ec69b5e887b4ed19e), [0b819b2](https://github.com/MKHC39/CYD-KSL-Translation/commit/0b819b2cf0a9e38fdade1c75cfc901922a5c84b4)
- Submodule `CYDProject`: Initial project structure and resource organization. [7116b84](https://github.com/MKHC39/CYDProject/commit/7116b84cbafec3d2bea9654da1479ac4fae6ddd7), [9c6777c](https://github.com/MKHC39/CYDProject/commit/9c6777c5841f198ece23768f042cbf6bcea8d8b7)
- Submodule `CYDProject`: Paper organization and additional resources. [7185593](https://github.com/MKHC39/CYDProject/commit/718559341902101588af71475589686aea5a978c)

