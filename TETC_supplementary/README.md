# Supplementary material for IEEE TETC

**From EM Pulses to Model Failure: Characterizing Physical Faults in Neural-Network Weight Memory**  
Jakub Breier, Štefan Kučerák, and Xiaolu Hou

This directory accompanies the revised manuscript and contains the derived experimental data and analysis scripts supporting its physical EMFI fault characterization. It is part of the [BP artifact repository](https://github.com/xkucerak/BP).

## Contents

### 1. Complete model-by-format breakdown

- [`data/campaign_by_model_dtype.csv`](data/campaign_by_model_dtype.csv): all 16 model × numerical-format strata, including BER, affected-byte fraction, and changed-bit multiplicity.
- [`data/campaign_summary.txt`](data/campaign_summary.txt): headline consistency checks and final-versus-nonfinal chunk comparison.

### 2. Complete per-address probability profile

- [`data/random_address_profile.npz`](data/random_address_profile.npz): arrays over all 4,194,304 byte addresses from 50 random-buffer injections:
  - `p_corrupt`: empirical probability of corruption for each byte address (float32).
  - `changed_count`: number of injections in which each byte address was corrupted (uint16).
  - `total_runs`: 50.
  - `groups`: identifiers of the five random buffers (`000`–`004`).
- [`data/address_bit_summary.csv`](data/address_bit_summary.csv) and [`data/joint_address_class_summary.csv`](data/joint_address_class_summary.csv): address-bit and joint-address-class summaries used in the manuscript.
- [`scripts/build_random_address_profile.py`](scripts/build_random_address_profile.py): profile-generation script.

### 3. Complete per-chunk physical fault load

- [`data/campaign_per_chunk.csv`](data/campaign_per_chunk.csv): statistics for all 352 clean/corrupted chunk pairs, including occupied size, BER, affected-byte fraction, changed-bit multiplicity, transition direction, normalized position, and final-chunk flag.
- [`scripts/analyze_campaign_stratified.py`](scripts/analyze_campaign_stratified.py): reproduction script for the campaign stratification.

Rates and probabilities in the CSV and NPZ files are fractions (multiply by 100 to express them as percentages).

## Raw experimental data

The large raw clean/corrupted model chunks are not duplicated in this directory. Consult the [main BP repository README](../README.md) for the artifact and its linked data store. Rebuilding the random-address profile also requires the five clean random buffers and their 50 corresponding post-injection dumps. These raw random-buffer files are not included in this supplementary directory.

## Usage and reproduction

Use Python 3.10 or later. From the `TETC_supplementary/` directory, install the dependencies:

```sh
python -m pip install -r requirements.txt
```

The original analysis environment's dependency versions are not recorded in the supplied supplementary files.

### Reproduce campaign statistics

Download and arrange the original binary files as follows (model names and formats are lowercase):

```text
BP/
  model_chunks/<model>/<model>_<dtype>/<chunk>.bin
  model_chunks_corr/<model>/<model>_<dtype>/<chunk>.bin
  TETC_supplementary/
```

From `TETC_supplementary/`, run:

```sh
python scripts/analyze_campaign_stratified.py --root .. --output-dir reproduced/campaign --strict
```

If the raw files are stored elsewhere, replace `..` with their root directory. A complete campaign should yield 352 rows in `campaign_per_chunk.csv` (excluding its header) and 16 rows in `campaign_by_model_dtype.csv`. The script also produces per-format, per-model, and floating-point exceedance summaries. It does not run neural-network inference. Check the reported chunk count even with `--strict`, since absent model/format directories are skipped by the script.

### Rebuild the random-address probability profile

The input directory must contain the clean files `000.bin` through `004.bin`, plus their matching corrupted dumps, for example `000_000.bin`, `000_001.bin`, and so on. The default group list covers all five clean buffers. For this campaign there should be ten corrupted dumps per clean buffer.

```sh
python scripts/build_random_address_profile.py --random-dir /path/to/random_buffers --output reproduced/random_address_profile.npz
```

Check that the script reports 4,194,304 addresses and 50 physical injections. The supplied script rebuilds the NPZ profile; the two address-class CSV summaries are provided as precomputed results and are not exported by either included script.

### Load the supplied address profile

```python
import numpy as np

with np.load("data/random_address_profile.npz", allow_pickle=False) as profile:
    p_corrupt = profile["p_corrupt"]
    changed_count = profile["changed_count"]
    total_runs = int(profile["total_runs"])
    print(p_corrupt.shape, total_runs)
```

Generated outputs are directed to `reproduced/` so that the original supplementary results remain unchanged.
