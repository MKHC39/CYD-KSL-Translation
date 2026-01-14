# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-14

### Added
- Initial project documentation and environment setup files: `README.md`, `environment.yml`, and `requirements.txt`. [9b0508f](https://github.com/MKHC39/CYD-KSL-Translation/commit/9b0508f9af1c0175ceae9926ea8abb54af2437c7)
- Translation scripts in `glossTL` directory (`check.py`, `pull_script.py`). [e563d33](https://github.com/MKHC39/CYD-KSL-Translation/commit/e563d3338dfc13ffee6a33e981299065439bd0dc)

### Changed
- Updated `glossTL/pull_script.py` with recent changes. [8f08eb9](https://github.com/MKHC39/CYD-KSL-Translation/commit/8f08eb9913b27b2361fef01ec00a7c4b090a986d)
- Refined validation logic in `glossTL/check.py`. [ed4029a](https://github.com/MKHC39/CYD-KSL-Translation/commit/ed4029a76dad3caea115e441fec5d90135c83e42)
- Updated `.gitignore` to exclude `preprocess` directory. [cd82892](https://github.com/MKHC39/CYD-KSL-Translation/commit/cd82892c20765c091d313f09d198d3def901c28f)

### Removed
- Redundant preprocessed KSL dataset files (`dev_info.npy`, `dev_manifest.jsonl`, etc.). [e563d33](https://github.com/MKHC39/CYD-KSL-Translation/commit/e563d3338dfc13ffee6a33e981299065439bd0dc)

