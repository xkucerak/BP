#!/usr/bin/env python3
"""
build_random_address_profile.py

Build an empirical per-address EMFI vulnerability profile from the five random
SRAM experiments:
    000.bin + 000_*.bin
    ...
    004.bin + 004_*.bin

The output .npz contains:
    p_corrupt      float32 per-address probability over all matched injections
    changed_count  uint16 number of injections in which each address changed
    total_runs     scalar number of injections
"""

from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-dir", type=Path, required=True,
                    help="directory containing 000.bin, 000_000.bin, ...")
    ap.add_argument("--groups", default="000,001,002,003,004")
    ap.add_argument("--output", type=Path, default=Path("random_address_profile.npz"))
    args = ap.parse_args()

    groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    counts = None
    total_runs = 0
    nbytes = None

    for g in groups:
        clean_path = args.random_dir / f"{g}.bin"
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        clean = np.memmap(clean_path, dtype=np.uint8, mode="r")
        if nbytes is None:
            nbytes = len(clean)
            counts = np.zeros(nbytes, dtype=np.uint16)
        elif len(clean) != nbytes:
            raise ValueError(f"Size mismatch for {clean_path}")

        faulty_files = sorted(args.random_dir.glob(f"{g}_*.bin"))
        if not faulty_files:
            raise FileNotFoundError(f"No faulty files for group {g}")

        print(f"{g}: {len(faulty_files)} faulty dumps")
        for fp in faulty_files:
            faulty = np.memmap(fp, dtype=np.uint8, mode="r")
            if len(faulty) != nbytes:
                raise ValueError(f"Size mismatch for {fp}")
            counts += np.asarray(faulty != clean, dtype=np.uint16)
            total_runs += 1

    p = counts.astype(np.float32) / float(total_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        p_corrupt=p,
        changed_count=counts,
        total_runs=np.array(total_runs, dtype=np.int32),
        groups=np.array(groups),
    )

    print(f"\nWrote: {args.output}")
    print(f"Address bytes: {nbytes:,}")
    print(f"Physical injections: {total_runs}")
    print(f"Mean per-address corruption probability: {100*p.mean():.4f}%")
    print(f"Never corrupted: {100*np.mean(counts == 0):.4f}%")
    print(f"Corrupted in every run: {100*np.mean(counts == total_runs):.4f}%")
    print(f"Non-zero-profile addresses: {np.count_nonzero(counts):,}")


if __name__ == "__main__":
    main()
