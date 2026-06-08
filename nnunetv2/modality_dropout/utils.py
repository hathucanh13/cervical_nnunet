"""
nnunetv2.modality_dropout.utils
────────────────────────────────
Shared helpers for the nnUNetMD pipeline.

Image I/O (NIfTI and NRRD) is supported via SimpleITK for the dataset
integrity check and zero-filled dataset creation.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ──────────────────────────────────────────────────────────────────────────────
# Image format helpers
# ──────────────────────────────────────────────────────────────────────────────

_NIFTI_EXTS = (".nii.gz", ".nii")
_NRRD_EXTS  = (".seg.nrrd", ".nrrd")
_ALL_EXTS   = _NIFTI_EXTS + _NRRD_EXTS


def collect_image_files(images_dir: Path) -> list[Path]:
    """Return all supported image files in images_dir, sorted."""
    files: list[Path] = []
    for ext in _ALL_EXTS:
        files.extend(images_dir.glob(f"*{ext}"))
    return sorted(set(files))


def strip_image_suffix(filename: str) -> str:
    """
    Remove both the format extension and the _XXXX channel suffix.
    'CC_001_0000.nii.gz'   → 'CC_001'
    'CC_001_0000.nrrd'     → 'CC_001'
    'CC_001_0000.seg.nrrd' → 'CC_001'
    """
    for ext in sorted(_ALL_EXTS, key=len, reverse=True):
        if filename.endswith(ext):
            return re.sub(r"_\d{4}$", "", filename[: -len(ext)])
    return re.sub(r"_\d{4}$", "", Path(filename).stem)


def sitk_load(path: Path) -> sitk.Image:
    """Load a NIfTI or NRRD image via SimpleITK."""
    return sitk.ReadImage(str(path))


def _get_file_ext(file_path: Path) -> str:
    """Return the image extension (.nii.gz, .nii, .nrrd, .seg.nrrd)."""
    for ext in sorted(_ALL_EXTS, key=len, reverse=True):
        if file_path.name.endswith(ext):
            return ext
    raise ValueError(f"Unsupported image extension: {file_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# nnU-Net environment
# ──────────────────────────────────────────────────────────────────────────────

def get_nnunet_base() -> tuple[Path, Path, Path]:
    raw = os.environ.get("nnUNet_raw")
    pre = os.environ.get("nnUNet_preprocessed")
    res = os.environ.get("nnUNet_results")
    missing = [k for k, v in [("nnUNet_raw", raw),
                               ("nnUNet_preprocessed", pre),
                               ("nnUNet_results", res)] if not v]
    if missing:
        raise EnvironmentError(
            f"Missing nnU-Net environment variable(s): {', '.join(missing)}"
        )
    return Path(raw), Path(pre), Path(res)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_dataset(identifier: str, raw_base: Path) -> tuple[int, str, Path]:
    """Accept numeric ID or full folder name. Returns (id, name, path)."""
    if identifier.isdigit():
        dataset_id = int(identifier)
        pattern    = re.compile(rf"^Dataset{dataset_id:03d}_")
        matches    = [d for d in raw_base.iterdir()
                      if d.is_dir() and pattern.match(d.name)]
        if not matches:
            raise FileNotFoundError(
                f"No dataset folder for ID {dataset_id} in {raw_base}"
            )
        folder = matches[0]
    else:
        folder = raw_base / identifier
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        m = re.match(r"^Dataset(\d+)_", folder.name)
        if not m:
            raise ValueError(
                f"'{folder.name}' does not follow the nnU-Net naming "
                "convention 'DatasetXXX_Name'."
            )
        dataset_id = int(m.group(1))

    return dataset_id, folder.name, folder


def _next_dataset_id(raw_base: Path) -> int:
    """Return the next available dataset ID (max existing + 1, minimum 1)."""
    existing_ids = []
    pattern = re.compile(r"^Dataset(\d+)_")
    for d in raw_base.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                existing_ids.append(int(m.group(1)))
    return max(existing_ids, default=0) + 1


def count_channels(dataset_json_path: Path) -> int:
    data = json.loads(dataset_json_path.read_text())
    return len(data.get("channel_names", data.get("modality", {})))


# ──────────────────────────────────────────────────────────────────────────────
# Zero-filled dataset creation
# ──────────────────────────────────────────────────────────────────────────────

def create_zerofilled_dataset(
    ss_raw: Path,
    mm_raw: Path,
    raw_base: Path,
    new_id: int | None = None,
) -> tuple[int, str, Path]:
    """
    Create a new nnU-Net raw dataset from the single-sequence dataset where
    every auxiliary channel (beyond ch0 / T2w) is a zero-filled image that
    matches the geometry of the anchor channel.

    The number of auxiliary channels to create is derived from the multimodal
    dataset's channel count so the resulting dataset has the same channel
    layout as the multimodal pretraining data.

    Directory layout produced
    ─────────────────────────
    Dataset{new_id}_ZF_{ss_name}/
        imagesTr/   ← _0000 copied from ss; _0001…_NNNN zero-filled
        labelsTr/   ← copied verbatim from ss
        imagesTs/   ← same treatment as imagesTr  (if present)
        dataset.json

    Parameters
    ──────────
    ss_raw      : Path to the single-sequence raw dataset folder.
    mm_raw      : Path to the multimodal raw dataset folder (to read channel count).
    raw_base    : nnUNet_raw root (used to find the next free dataset ID).
    new_id      : Override the auto-assigned dataset ID (optional).

    Returns
    ───────
    (new_dataset_id, new_dataset_name, new_dataset_path)
    """
    n_ch_mm = count_channels(mm_raw / "dataset.json")
    n_ch_ss = count_channels(ss_raw / "dataset.json")

    if n_ch_ss != 1:
        raise ValueError(
            f"Expected the single-sequence dataset to have exactly 1 channel, "
            f"got {n_ch_ss}.  Only T2w-only datasets are supported."
        )
    if n_ch_mm <= n_ch_ss:
        raise ValueError(
            f"Multimodal channel count ({n_ch_mm}) must be greater than "
            f"single-sequence channel count ({n_ch_ss})."
        )

    n_zero_channels = n_ch_mm - n_ch_ss   # number of channels to zero-fill

    # Determine new dataset ID and folder name
    if new_id is None:
        new_id = _next_dataset_id(raw_base)

    # Build a short suffix from the single-seq folder name (strip DatasetXXX_ prefix)
    ss_suffix = re.sub(r"^Dataset\d+_", "", ss_raw.name)
    new_name  = f"Dataset{new_id:03d}_ZF_{ss_suffix}"
    new_path  = raw_base / new_name

    if new_path.exists():
        print(f"[nnUNetMD] Zero-filled dataset already exists: {new_path}")
        print( "[nnUNetMD] Delete it manually to recreate, or use --skip-create-zf.")
        return new_id, new_name, new_path

    print(f"\n[nnUNetMD] Creating zero-filled dataset: {new_name}")
    print(f"  Source             : {ss_raw.name}")
    print(f"  Anchor channel     : 0  (T2w — copied verbatim)")
    print(f"  Zero-filled chans  : {n_ch_ss}–{n_ch_mm - 1}  ({n_zero_channels} channel(s))")

    new_path.mkdir(parents=True)
    (new_path / "imagesTr").mkdir()
    (new_path / "labelsTr").mkdir()

    # ── Copy labels (unchanged) ───────────────────────────────────────────────
    for label_file in sorted((ss_raw / "labelsTr").iterdir()):
        shutil.copy2(label_file, new_path / "labelsTr" / label_file.name)

    # ── Process image splits ──────────────────────────────────────────────────
    splits_to_process = [("imagesTr", True)]
    if (ss_raw / "imagesTs").exists():
        (new_path / "imagesTs").mkdir()
        splits_to_process.append(("imagesTs", False))

    for split_dir_name, verbose in splits_to_process:
        src_dir = ss_raw / split_dir_name
        dst_dir = new_path / split_dir_name

        anchor_files = collect_image_files(src_dir)
        # Keep only channel-0 files (the T2w anchor)
        anchor_files = [f for f in anchor_files
                        if re.search(r"_0000\.", f.name)]

        if verbose:
            print(f"  Processing {split_dir_name}: {len(anchor_files)} case(s)")

        for anchor_path in anchor_files:
            ext       = _get_file_ext(anchor_path)
            case_id   = strip_image_suffix(anchor_path.name)

            # Copy anchor channel
            dst_anchor = dst_dir / f"{case_id}_0000{ext}"
            shutil.copy2(anchor_path, dst_anchor)

            # Create zero-filled images for auxiliary channels
            ref_img   = sitk_load(anchor_path)
            zero_arr  = np.zeros(sitk.GetArrayFromImage(ref_img).shape,
                                 dtype=np.float32)
            zero_img  = sitk.GetImageFromArray(zero_arr)
            zero_img.CopyInformation(ref_img)   # preserve spacing/origin/direction

            for ch_idx in range(n_ch_ss, n_ch_mm):
                dst_path = dst_dir / f"{case_id}_{ch_idx:04d}{ext}"
                sitk.WriteImage(zero_img, str(dst_path))

    # ── Build dataset.json ────────────────────────────────────────────────────
    ss_meta = json.loads((ss_raw / "dataset.json").read_text())
    mm_meta = json.loads((mm_raw / "dataset.json").read_text())

    # Reconstruct channel_names: anchor from ss, rest from mm
    ss_channels = ss_meta.get("channel_names", ss_meta.get("modality", {}))
    mm_channels = mm_meta.get("channel_names", mm_meta.get("modality", {}))

    # Normalise to str-keyed dict
    ss_channels = {str(k): v for k, v in ss_channels.items()}
    mm_channels = {str(k): v for k, v in mm_channels.items()}

    new_channels: dict[str, str] = {}
    for i in range(n_ch_mm):
        key = str(i)
        if i < n_ch_ss:
            new_channels[key] = ss_channels.get(key, f"ch{i}")
        else:
            # Label the zero-filled channel after the mm modality if available,
            # with a clear suffix so it's obvious these are synthetic zeros.
            mm_label = mm_channels.get(key, f"ch{i}")
            new_channels[key] = f"{mm_label}_zerofilled"

    new_meta = {
        **ss_meta,
        "channel_names": new_channels,
        "name"         : new_name,
        # Record provenance so downstream tools can validate
        "nnUNetMD_source_single_seq" : ss_raw.name,
        "nnUNetMD_source_multimodal" : mm_raw.name,
        "nnUNetMD_zerofilled_channels": list(range(n_ch_ss, n_ch_mm)),
    }
    # Remove old 'modality' key if present (replaced by channel_names)
    new_meta.pop("modality", None)

    (new_path / "dataset.json").write_text(json.dumps(new_meta, indent=2))

    print(f"  Channel names      : {new_channels}")
    print(f"  dataset.json saved : {new_path / 'dataset.json'}")
    print(f"[nnUNetMD] Zero-filled dataset ready → {new_path}")
    return new_id, new_name, new_path


# ──────────────────────────────────────────────────────────────────────────────
# Metadata  (nnUNetMD_metadata.json lives in nnUNet_preprocessed/<mm_dataset>/)
# ──────────────────────────────────────────────────────────────────────────────

METADATA_FILENAME = "nnUNetMD_metadata.json"


def write_metadata(preprocessed_base: Path, multimodal_folder: str,
                   single_seq_folder: str, n_channels_multimodal: int,
                   n_channels_single: int,
                   zerofilled_folder: str | None = None,
                   zerofilled_id: int | None = None) -> Path:
    meta = {
        "multimodal_dataset"      : multimodal_folder,
        "single_sequence_dataset" : single_seq_folder,
        "zerofilled_dataset"      : zerofilled_folder,   # new: disk-based ZF dataset
        "zerofilled_dataset_id"   : zerofilled_id,
        "n_channels_multimodal"   : n_channels_multimodal,
        "n_channels_single"       : n_channels_single,
        "pretrain_checkpoint"     : None,
        "pretrain_complete"       : False,
    }
    path = preprocessed_base / multimodal_folder / METADATA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
    print(f"[nnUNetMD] Metadata written → {path}")
    return path


def load_metadata(preprocessed_base: Path, multimodal_folder: str) -> dict:
    path = preprocessed_base / multimodal_folder / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {path}.\n"
            "Run 'nnUNetMD_plan_and_preprocess' first."
        )
    return json.loads(path.read_text())


def update_metadata(preprocessed_base: Path, multimodal_folder: str,
                    **kwargs) -> None:
    meta = load_metadata(preprocessed_base, multimodal_folder)
    meta.update(kwargs)
    path = preprocessed_base / multimodal_folder / METADATA_FILENAME
    path.write_text(json.dumps(meta, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Plans helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_plans(preprocessed_base: Path, folder_name: str,
               plans_name: str = "nnUNetPlans") -> dict:
    path = preprocessed_base / folder_name / f"{plans_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Plans not found: {path}")
    return json.loads(path.read_text())


def save_plans(preprocessed_base: Path, folder_name: str,
               plans: dict, plans_name: str = "nnUNetPlans_MD") -> None:
    path = preprocessed_base / folder_name / f"{plans_name}.json"
    path.write_text(json.dumps(plans, indent=2))
    print(f"[nnUNetMD] Plans saved → {path}")


def patch_multimodal_plans(mm_plans: dict, ss_plans: dict,
                           n_channels_mm: int,
                           n_real_channels: int | None = None) -> dict:
    """
    Replace the 3d_fullres block in multimodal plans with the one from the
    single-sequence plans, then set normalization schemes per channel:

      ch0 … ch(n_real_channels-1) : ZScoreNormalization  (real image data)
      ch(n_real_channels) … chN-1 : NoNormalization       (zero-filled channels)

    Used in two contexts:

    1. Multimodal dataset (plan_and_preprocess):
         n_real_channels = n_channels_mm  (all channels are real)
         → ZScoreNormalization on every channel

    2. Zero-filled Stage-2 dataset (train_from_pretrain):
         n_real_channels = n_channels_single  (only the anchor T2w channel is real)
         → ZScoreNormalization on ch0 only; NoNormalization on ch1+
         The zero-filled channels contain exactly 0.0 on disk; NoNormalization
         preserves those zeros rather than producing degenerate ZScore output.
    """
    import copy

    if n_real_channels is None:
        n_real_channels = n_channels_mm

    if n_real_channels > n_channels_mm:
        raise ValueError(
            f"n_real_channels ({n_real_channels}) cannot exceed "
            f"n_channels_mm ({n_channels_mm})."
        )

    patched  = copy.deepcopy(mm_plans)
    ss_block = copy.deepcopy(ss_plans["configurations"]["3d_fullres"])
    patched["configurations"]["3d_fullres"] = ss_block

    base_scheme = ss_block.get("normalization_schemes", ["ZScoreNormalization"])[0]

    patched["configurations"]["3d_fullres"]["normalization_schemes"] = (
        [base_scheme]         * n_real_channels
        + ["NoNormalization"] * (n_channels_mm - n_real_channels)
    )
    patched["configurations"]["3d_fullres"]["use_mask_for_norm"] = (
        [False] * n_channels_mm
    )
    return patched


# ──────────────────────────────────────────────────────────────────────────────
# Dataset integrity check
# ──────────────────────────────────────────────────────────────────────────────

def verify_dataset_integrity(raw_dataset_path: Path) -> bool:
    """
    Verify that all images in imagesTr have consistent shapes.
    Supports NIfTI and NRRD via SimpleITK.

    - Within a case: all channels must share the same shape → ERROR if not.
    - Across cases:  shape differences are allowed (nnU-Net resamples) → WARNING.
    Returns True when no critical errors are found.
    """
    images_dir = raw_dataset_path / "imagesTr"
    if not images_dir.exists():
        raise FileNotFoundError(f"imagesTr not found in {raw_dataset_path}")

    all_files = collect_image_files(images_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No image files found in {images_dir}  (supported: {_ALL_EXTS})"
        )

    detected_exts = sorted(
        {ext for f in all_files for ext in _ALL_EXTS if f.name.endswith(ext)},
        key=len, reverse=True,
    )
    print(f"\n[nnUNetMD] Verifying: {raw_dataset_path.name}")
    print(f"  Files    : {len(all_files)}  |  Format(s): {detected_exts}")

    cases: dict[str, list[Path]] = defaultdict(list)
    for f in all_files:
        cases[strip_image_suffix(f.name)].append(f)

    all_ok       = True
    case_shapes  : dict[str, tuple] = {}

    for case_id, files in sorted(cases.items()):
        shapes = []
        for f in sorted(files):
            try:
                shapes.append(tuple(sitk_load(f).GetSize()))
            except Exception as e:
                print(f"  [ERROR] Cannot load {f.name}: {e}")
                all_ok = False

        if len(set(shapes)) > 1:
            print(f"  [ERROR] '{case_id}': channel shape mismatch!")
            for f, s in zip(sorted(files), shapes):
                print(f"    {f.name}: {s}")
            all_ok = False
        elif shapes:
            case_shapes[case_id] = shapes[0]

    unique = set(case_shapes.values())
    if len(unique) > 1:
        print("  [WARNING] Cases have different spatial shapes "
              "(nnU-Net will resample — check orientations are consistent).")
        counts: dict[tuple, list] = {}
        for cid, sh in case_shapes.items():
            counts.setdefault(sh, []).append(cid)
        for sh, cids in sorted(counts.items()):
            print(f"    {sh}: {len(cids)} case(s), e.g. {cids[0]}")
    elif unique:
        print(f"  [OK] Uniform spatial shape: {next(iter(unique))}")

    status = "[OK] Passed" if all_ok else "[FAIL] Errors found — fix before continuing"
    print(f"  {status}\n")
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ──────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], description: str = "") -> None:
    print(f"\n[nnUNetMD] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): "
            f"{description or cmd[0]}"
        )