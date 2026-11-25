#!/usr/bin/env python3
"""
analyze_campaign_stratified.py

Analyze the full physical clean/corrupted chunk campaign from the BP artifact.

Expected layout:
ROOT/
  model_chunks/<model>/<model>_<dtype>/<chunk>.bin
  model_chunks_corr/<model>/<model>_<dtype>/<chunk>.bin

Outputs:
  campaign_per_chunk.csv
  campaign_by_model_dtype.csv
  campaign_by_dtype.csv
  campaign_by_model.csv
  fp_exceedance_per_chunk.csv
  campaign_summary.txt

The script computes:
  - occupied chunk size
  - BER
  - corrupted-byte fraction
  - changed bits per corrupted byte
  - 0->1 fraction
  - normalized chunk position
  - final-chunk flag
  - stratified means / byte-weighted totals
  - robust floating-point exceedance statistics relative to each clean
    chunk's maximum finite |weight|, plus NaN/Inf fractions.

No inference is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

MODELS = ("resnet18", "resnet34", "resnet50", "vgg11")
DTYPES = ("fp32", "fp16", "int8", "int4")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="BP artifact root containing model_chunks/ and model_chunks_corr/")
    p.add_argument("--output-dir", type=Path, default=Path("campaign_stratified"))
    p.add_argument("--strict", action="store_true",
                   help="fail if any expected clean/corrupted counterpart is missing")
    return p.parse_args()


def paired_chunks(root: Path):
    clean_root = root / "model_chunks"
    corr_root = root / "model_chunks_corr"

    for model in MODELS:
        for dtype in DTYPES:
            sub = f"{model}_{dtype}"
            cdir = clean_root / model / sub
            fdir = corr_root / model / sub

            if not cdir.exists() or not fdir.exists():
                continue

            clean_files = {p.stem: p for p in cdir.glob("*.bin") if p.stem.isdigit()}
            corr_files = {p.stem: p for p in fdir.glob("*.bin") if p.stem.isdigit()}
            ids = sorted(set(clean_files) | set(corr_files), key=int)

            for cid in ids:
                cp = clean_files.get(cid)
                fp = corr_files.get(cid)
                yield model, dtype, cid, cp, fp


def chunk_stats(clean: np.ndarray, corr: np.ndarray):
    if clean.size != corr.size:
        raise ValueError(f"size mismatch: {clean.size} vs {corr.size}")

    xor = np.bitwise_xor(clean, corr)
    changed = xor != 0
    per_byte = POPCOUNT[xor]

    nbytes = int(clean.size)
    changed_bytes = int(changed.sum())
    changed_bits = int(per_byte.sum())
    total_bits = nbytes * 8

    zto = 0
    otz = 0
    for bit in range(8):
        mask = np.uint8(1 << bit)
        cb = (clean & mask) != 0
        fb = (corr & mask) != 0
        zto += int((~cb & fb).sum())
        otz += int((cb & ~fb).sum())

    return {
        "occupied_bytes": nbytes,
        "changed_bytes": changed_bytes,
        "changed_bits": changed_bits,
        "affected_byte_fraction": changed_bytes / nbytes if nbytes else np.nan,
        "ber": changed_bits / total_bits if total_bits else np.nan,
        "bits_per_changed_byte": changed_bits / changed_bytes if changed_bytes else 0.0,
        "zero_to_one": zto,
        "one_to_zero": otz,
        "zero_to_one_fraction": zto / changed_bits if changed_bits else np.nan,
    }


def fp_stats(clean_u8: np.ndarray, corr_u8: np.ndarray, dtype: str):
    npdtype = np.dtype("<f4") if dtype == "fp32" else np.dtype("<f2")
    itemsize = npdtype.itemsize
    usable = (clean_u8.size // itemsize) * itemsize

    clean = clean_u8[:usable].tobytes()
    corr = corr_u8[:usable].tobytes()
    clean_f = np.frombuffer(clean, dtype=npdtype)
    corr_f = np.frombuffer(corr, dtype=npdtype)

    clean_finite = np.isfinite(clean_f)
    corr_finite = np.isfinite(corr_f)

    if not np.any(clean_finite):
        clean_abs_max = np.nan
    else:
        clean_abs_max = float(np.max(np.abs(clean_f[clean_finite]).astype(np.float64)))

    out = {
        "weights_in_chunk": int(clean_f.size),
        "clean_abs_max": clean_abs_max,
        "corr_nan_fraction": float(np.mean(np.isnan(corr_f))),
        "corr_inf_fraction": float(np.mean(np.isinf(corr_f))),
        "corr_nonfinite_fraction": float(np.mean(~corr_finite)),
    }

    abs_corr = np.abs(corr_f.astype(np.float64))
    for k in (10, 100, 1000):
        if not np.isfinite(clean_abs_max) or clean_abs_max <= 0:
            frac = np.nan
        else:
            frac = float(np.mean(corr_finite & (abs_corr > k * clean_abs_max)))
        out[f"finite_gt_{k}x_cleanmax_fraction"] = frac

    # Also give the fraction above fixed fractions of the format maximum.
    # Useful if the clean maximum varies substantially across chunks.
    fmt_max = np.finfo(npdtype).max
    for fracmax in (1e-2, 1e-1, 0.5):
        out[f"finite_gt_{fracmax:g}_formatmax_fraction"] = float(
            np.mean(corr_finite & (abs_corr > fracmax * fmt_max))
        )

    return out


def aggregate(df: pd.DataFrame, keys: list[str]):
    rows = []
    for vals, g in df.groupby(keys, dropna=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        d = dict(zip(keys, vals))

        total_bytes = int(g["occupied_bytes"].sum())
        total_bits_capacity = total_bytes * 8
        total_changed_bytes = int(g["changed_bytes"].sum())
        total_changed_bits = int(g["changed_bits"].sum())

        d.update({
            "n_chunks": len(g),
            "total_occupied_bytes": total_bytes,

            # Simple per-chunk means: directly expose composition bias.
            "mean_chunk_ber": g["ber"].mean(),
            "std_chunk_ber": g["ber"].std(ddof=1) if len(g) > 1 else 0.0,
            "mean_chunk_affected_byte_fraction": g["affected_byte_fraction"].mean(),
            "std_chunk_affected_byte_fraction": (
                g["affected_byte_fraction"].std(ddof=1) if len(g) > 1 else 0.0
            ),
            "mean_chunk_bits_per_changed_byte": g["bits_per_changed_byte"].mean(),

            # Byte-weighted global rates.
            "byte_weighted_ber": (
                total_changed_bits / total_bits_capacity if total_bits_capacity else np.nan
            ),
            "byte_weighted_affected_byte_fraction": (
                total_changed_bytes / total_bytes if total_bytes else np.nan
            ),
            "byte_weighted_bits_per_changed_byte": (
                total_changed_bits / total_changed_bytes if total_changed_bytes else np.nan
            ),
        })
        rows.append(d)
    return pd.DataFrame(rows)


def pct(x):
    return 100.0 * x


def main():
    args = parse_args()
    root = args.root.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    fp_rows = []
    missing = []

    groups = {}
    for model, dtype, cid, cp, fp in paired_chunks(root):
        groups.setdefault((model, dtype), []).append((cid, cp, fp))

    for (model, dtype), entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda x: int(x[0]))
        n_chunks = len(entries)

        for ordinal, (cid, cp, fp) in enumerate(entries):
            if cp is None or fp is None:
                missing.append((model, dtype, cid, str(cp), str(fp)))
                continue

            clean = np.fromfile(cp, dtype=np.uint8)
            corr = np.fromfile(fp, dtype=np.uint8)
            s = chunk_stats(clean, corr)

            row = {
                "model": model,
                "dtype": dtype,
                "chunk_id": cid,
                "chunk_index": ordinal,
                "n_chunks_in_model_dtype": n_chunks,
                "normalized_position": ordinal / (n_chunks - 1) if n_chunks > 1 else 0.0,
                "is_final_chunk": ordinal == (n_chunks - 1),
                **s,
            }
            rows.append(row)

            if dtype in {"fp32", "fp16"}:
                fs = fp_stats(clean, corr, dtype)
                fp_rows.append({
                    "model": model,
                    "dtype": dtype,
                    "chunk_id": cid,
                    "normalized_position": row["normalized_position"],
                    "is_final_chunk": row["is_final_chunk"],
                    "occupied_bytes": row["occupied_bytes"],
                    "ber": row["ber"],
                    "affected_byte_fraction": row["affected_byte_fraction"],
                    **fs,
                })

            print(
                f"{model:8s} {dtype:4s} {cid} "
                f"{s['occupied_bytes']/2**20:6.3f} MiB  "
                f"BER={pct(s['ber']):6.3f}%  "
                f"bytes={pct(s['affected_byte_fraction']):6.3f}%"
            )

    if missing and args.strict:
        raise RuntimeError(f"{len(missing)} missing chunk counterparts: {missing[:10]}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No clean/corrupted chunk pairs found. Check --root.")

    df = df.sort_values(["model", "dtype", "chunk_index"]).reset_index(drop=True)
    df.to_csv(outdir / "campaign_per_chunk.csv", index=False)

    by_md = aggregate(df, ["model", "dtype"])
    by_dtype = aggregate(df, ["dtype"])
    by_model = aggregate(df, ["model"])

    by_md.to_csv(outdir / "campaign_by_model_dtype.csv", index=False)
    by_dtype.to_csv(outdir / "campaign_by_dtype.csv", index=False)
    by_model.to_csv(outdir / "campaign_by_model.csv", index=False)

    fpdf = pd.DataFrame(fp_rows)
    if not fpdf.empty:
        fpdf.to_csv(outdir / "fp_exceedance_per_chunk.csv", index=False)

    # Whole-campaign simple mean and byte-weighted rates.
    total_bytes = int(df["occupied_bytes"].sum())
    total_changed_bytes = int(df["changed_bytes"].sum())
    total_changed_bits = int(df["changed_bits"].sum())

    mean_ber = float(df["ber"].mean())
    mean_aff = float(df["affected_byte_fraction"].mean())
    byte_ber = total_changed_bits / (8 * total_bytes)
    byte_aff = total_changed_bytes / total_bytes

    final = df[df["is_final_chunk"]]
    nonfinal = df[~df["is_final_chunk"]]

    lines = []
    lines.append(f"Paired chunks: {len(df)}")
    lines.append(f"Missing counterparts skipped: {len(missing)}")
    lines.append("")
    lines.append("WHOLE CAMPAIGN")
    lines.append(f"simple mean BER across chunks: {pct(mean_ber):.4f}%")
    lines.append(f"simple mean affected bytes across chunks: {pct(mean_aff):.4f}%")
    lines.append(f"byte-weighted BER: {pct(byte_ber):.4f}%")
    lines.append(f"byte-weighted affected bytes: {pct(byte_aff):.4f}%")
    lines.append("")
    lines.append("CHUNK-SIZE CONFOUND")
    lines.append(
        f"non-final chunks: n={len(nonfinal)}, "
        f"mean BER={pct(nonfinal['ber'].mean()):.4f}%, "
        f"mean affected bytes={pct(nonfinal['affected_byte_fraction'].mean()):.4f}%"
    )
    lines.append(
        f"final chunks: n={len(final)}, "
        f"mean BER={pct(final['ber'].mean()):.4f}%, "
        f"mean affected bytes={pct(final['affected_byte_fraction'].mean()):.4f}%"
    )

    lines.append("")
    lines.append("BY DTYPE (simple chunk mean / byte-weighted)")
    for _, r in by_dtype.sort_values("dtype").iterrows():
        lines.append(
            f"{r['dtype']:4s}: n={int(r['n_chunks']):3d}; "
            f"BER={pct(r['mean_chunk_ber']):.4f}% / {pct(r['byte_weighted_ber']):.4f}%; "
            f"affected={pct(r['mean_chunk_affected_byte_fraction']):.4f}% / "
            f"{pct(r['byte_weighted_affected_byte_fraction']):.4f}%"
        )

    lines.append("")
    lines.append("BY MODEL x DTYPE")
    for _, r in by_md.sort_values(["model", "dtype"]).iterrows():
        lines.append(
            f"{r['model']:8s} {r['dtype']:4s}: n={int(r['n_chunks']):3d}; "
            f"mean BER={pct(r['mean_chunk_ber']):.4f}%; "
            f"mean affected={pct(r['mean_chunk_affected_byte_fraction']):.4f}%; "
            f"bits/affected-byte={r['mean_chunk_bits_per_changed_byte']:.3f}"
        )

    summary = "\n".join(lines) + "\n"
    (outdir / "campaign_summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
