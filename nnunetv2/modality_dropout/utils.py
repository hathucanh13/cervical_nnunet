"""
nnunetv2.modality_dropout.utils
────────────────────────────────
Shared helpers for the nnUNetMD pipeline.

Image I/O (NIfTI and NRRD) is supported via SimpleITK for the dataset
integrity check only.  Zero-filled channel creation is no longer needed —
auxiliary channels are zero-padded on the fly by nnUNetTrainerStage2.
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import SimpleITK as sitk


# ──────────────────────────────────────────────────────────────────────────────
# Image format helpers  (used by verify_dataset_integrity only)
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


def count_channels(dataset_json_path: Path) -> int:
    data = json.loads(dataset_json_path.read_text())
    return len(data.get("channel_names", data.get("modality", {})))


# ──────────────────────────────────────────────────────────────────────────────
# Metadata  (nnUNetMD_metadata.json lives in nnUNet_preprocessed/<mm_dataset>/)
# ──────────────────────────────────────────────────────────────────────────────

METADATA_FILENAME = "nnUNetMD_metadata.json"


def write_metadata(preprocessed_base: Path, multimodal_folder: str,
                   single_seq_folder: str, n_channels_multimodal: int,
                   n_channels_single: int) -> Path:
    meta = {
        "multimodal_dataset"     : multimodal_folder,
        "single_sequence_dataset": single_seq_folder,
        "n_channels_multimodal"  : n_channels_multimodal,
        "n_channels_single"      : n_channels_single,
        "pretrain_checkpoint"    : None,
        "pretrain_complete"      : False,
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
      ch(n_real_channels) … chN-1 : NoNormalization       (auxiliary channels)

    Used in two contexts:

    1. Multimodal dataset (plan_and_preprocess):
         n_real_channels = n_channels_mm  (all channels are real)
         → ZScoreNormalization on every channel

    2. Single-sequence Stage-2 dataset (train_from_pretrain):
         n_real_channels = n_channels_single  (only the anchor channel is real)
         → ZScoreNormalization on ch0 only; NoNormalization on ch1+
         The auxiliary channels are not present on disk — they are zero-padded
         on the fly by nnUNetTrainerStage2 at batch load time.
         NoNormalization ensures that if zeros ever reach the preprocessor
         (e.g. future disk-based fallback), they are left at exactly 0.0
         rather than producing degenerate ZScore output.
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

    all_ok      = True
    case_shapes : dict[str, tuple] = {}

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