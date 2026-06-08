"""
nnUNetMD_train_from_pretrain  —  entry point
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import SimpleITK as sitk

from nnunetv2.modality_dropout.utils import (
    collect_image_files,
    count_channels,
    get_file_ending,
    get_nnunet_base,
    load_metadata,
    load_plans,
    resolve_dataset,
    run_cmd,
    save_plans,
    sitk_load,
    sitk_save,
    sitk_zeros_like,
    strip_image_suffix,
    update_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nnUNetMD_train_from_pretrain",
        description=(
            "Stage 2: load pretrained multimodal weights and run standard "
            "nnU-Net cross-validation on the single-sequence dataset.\n"
            "Run 'nnUNetMD_pretrain' first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset_multimodal")
    p.add_argument("dataset_single_sequence")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--trainer", default="nnUNetTrainerStage2")
    p.add_argument("--plans-name", default="nnUNetPlans")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (default: read from metadata).")
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--skip-prepare", action="store_true",
                   help="Skip dataset preparation if already done in a previous run.")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# 3-channel Stage-2 dataset preparation
# ──────────────────────────────────────────────────────────────────────────────
def _next_free_dataset_id(raw_base: Path) -> int:
    """
    Scan nnUNet_raw for existing DatasetXXX folders and return the next
    unused numeric ID (max existing + 1, minimum 100 to avoid clashing
    with standard benchmark datasets).
    """
    existing = []
    for d in raw_base.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"^Dataset(\d+)_", d.name)
        if m:
            existing.append(int(m.group(1)))
    return max(existing, default=99) + 1

def _prepare_stage2_dataset(ss_raw: Path, raw_base: Path,
                             n_ch_mm: int, n_ch_ss: int) -> tuple[int, str, Path]:
    """
    Create DatasetXXX_<name>_MD with a NEW unique numeric ID:
      ch0       = copied from original single-sequence (_0000)
      ch1…chN-1 = zero-filled in the SAME format as the source

    A new ID is assigned (not the source ID) to avoid nnU-Net's dataset
    resolver finding two folders with the same ID prefix.
    """
    file_ending = get_file_ending(ss_raw / "dataset.json")
    print(f"  File format : {file_ending}")

    m        = re.match(r"^(Dataset\d+_.+?)(_MD)?$", ss_raw.name)
    # Strip the DatasetXXX_ prefix to get the bare name, e.g. "CC"
    bare     = re.sub(r"^Dataset\d+_", "", m.group(1) if m else ss_raw.name)

    # Assign a new free ID so DatasetXXX_<bare>_MD never collides
    new_id   = _next_free_dataset_id(raw_base)
    new_name = f"Dataset{new_id:03d}_{bare}_MD"
    new_path = raw_base / new_name

    print(f"\n[nnUNetMD] Preparing Stage-2 dataset: {new_name}  (ID={new_id})")
    print(f"  Source   : {ss_raw.name}")
    print(f"  Channels : 1 real + {n_ch_mm - 1} zero-filled")

    (new_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (new_path / "labelsTr").mkdir(parents=True, exist_ok=True)

    # Collect only _0000 anchors
    src_images = [
        f for f in collect_image_files(ss_raw / "imagesTr")
        if f.name.replace(file_ending, "").endswith("_0000")
    ]
    if not src_images:
        raise FileNotFoundError(
            f"No _0000 images found in {ss_raw / 'imagesTr'} "
            f"with extension '{file_ending}'"
        )

    for src in src_images:
        case_id = strip_image_suffix(src.name)

        # ch0: verbatim copy
        shutil.copy2(src, new_path / "imagesTr" / f"{case_id}_0000{file_ending}")

        # ch1+: zero-filled, same format
        ref  = sitk_load(src)
        zero = sitk_zeros_like(ref)
        for ch in range(1, n_ch_mm):
            sitk_save(zero, new_path / "imagesTr" / f"{case_id}_{ch:04d}{file_ending}")

    # Copy labels (format-agnostic)
    for seg in collect_image_files(ss_raw / "labelsTr"):
        shutil.copy2(seg, new_path / "labelsTr" / seg.name)

    # Update dataset.json
    orig              = json.loads((ss_raw / "dataset.json").read_text())
    orig_ch           = orig.get("channel_names", orig.get("modality", {}))
    new_ch            = dict(orig_ch)
    for ch in range(n_ch_ss, n_ch_mm):
        new_ch[str(ch)] = f"zero_ch{ch}"
    orig["channel_names"] = new_ch
    orig.pop("modality", None)
    orig["file_ending"]   = file_ending
    orig["numTraining"]   = len(src_images)
    (new_path / "dataset.json").write_text(json.dumps(orig, indent=2))

    print(f"  [OK] {len(src_images)} cases written to {new_path}")
    return new_id, new_name, new_path


def _prepare_stage2_plans(pre_base: Path, mm_name: str,
                           new_ss_name: str, n_ch_mm: int,
                           plans_name: str) -> None:
    """
    Copy multimodal plans to Stage-2 dataset with corrected normalization:
      ch0 : ZScoreNormalization  (real T2W data)
      ch1+: NoNormalization      (zeros must stay exactly 0.0)
    """
    import copy
    plans = copy.deepcopy(load_plans(pre_base, mm_name, plans_name))
    plans["dataset_name"] = new_ss_name
    plans["configurations"]["3d_fullres"]["normalization_schemes"] = (
        ["ZScoreNormalization"] + ["NoNormalization"] * (n_ch_mm - 1)
    )
    plans["configurations"]["3d_fullres"]["use_mask_for_norm"] = (
        [False] * n_ch_mm
    )
    (pre_base / new_ss_name).mkdir(parents=True, exist_ok=True)
    save_plans(pre_base, new_ss_name, plans, plans_name)
    cfg = plans["configurations"]["3d_fullres"]
    print(f"  normalization_schemes : {cfg['normalization_schemes']}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raw_base, pre_base, res_base = get_nnunet_base()

    mm_id, mm_name, mm_raw = resolve_dataset(args.dataset_multimodal,      raw_base)
    ss_id, ss_name, ss_raw = resolve_dataset(args.dataset_single_sequence, raw_base)

    # Verify pretrain is complete
    meta = load_metadata(pre_base, mm_name)
    if not meta.get("pretrain_complete", False):
        print(f"[nnUNetMD] ERROR: pretrain not complete. "
              f"Run: nnUNetMD_pretrain {args.dataset_multimodal} "
              f"{args.dataset_single_sequence}")
        return 1

    checkpoint = args.checkpoint or meta.get("pretrain_checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        print(f"[nnUNetMD] ERROR: checkpoint not found: {checkpoint}")
        return 1

    n_ch_mm = meta["n_channels_multimodal"]
    n_ch_ss = meta["n_channels_single"]

    print(f"\n[nnUNetMD] Stage 2 fine-tuning")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Trainer    : {args.trainer}")
    print(f"  Folds      : {args.folds}")

    # Step 1: Prepare 3-channel Stage-2 dataset
    if not args.skip_prepare:
        new_ss_id, new_ss_name, _ = _prepare_stage2_dataset(
            ss_raw, raw_base, n_ch_mm, n_ch_ss)

        print(f"\n[nnUNetMD] Copying plans → {new_ss_name}")
        _prepare_stage2_plans(pre_base, mm_name, new_ss_name,
                               n_ch_mm, args.plans_name)

        print(f"\n[nnUNetMD] Preprocessing {new_ss_name} …")
        run_cmd(["nnUNetv2_preprocess",
                 "-d", str(new_ss_id),
                 "-c", "3d_fullres",
                 "--plans_name", args.plans_name,
                 "-np", str(args.num_processes)],
                f"preprocess {new_ss_name}")

        update_metadata(pre_base, mm_name,
                        stage2_dataset=new_ss_name,
                        stage2_dataset_id=new_ss_id)
    else:
        new_ss_name = meta.get("stage2_dataset")
        new_ss_id   = meta.get("stage2_dataset_id")
        if not new_ss_name:
            print("[nnUNetMD] ERROR: --skip-prepare used but 'stage2_dataset' "
                  "not in metadata. Run without --skip-prepare first.")
            return 1
        print(f"  [skip] Using previously prepared dataset: {new_ss_name}")

    # Step 2: Cross-validation
    print(f"\n[nnUNetMD] Cross-validation on {new_ss_name}")
    failed = []
    for fold in args.folds:
        print(f"\n[nnUNetMD] ── fold {fold} ──")
        try:
            run_cmd(
                ["nnUNetv2_train",
                 str(new_ss_name), "3d_fullres", str(fold),
                 "-tr",    args.trainer,
                 "-p",     args.plans_name,
                 "-pretrained_weights", checkpoint,
                 "--npz"],
                f"Stage 2 fold {fold}",
            )
        except RuntimeError as e:
            print(f"[nnUNetMD] WARNING fold {fold} failed: {e}")
            failed.append(fold)

    # Step 3: Find best configuration when all folds complete
    if not failed:
        print("\n[nnUNetMD] Finding best configuration …")
        try:
            run_cmd(
                ["nnUNetv2_find_best_configuration",
                 str(new_ss_name),
                 "-c", "3d_fullres",
                 "-tr", args.trainer,
                 "-p", args.plans_name],
                "find_best_configuration",
            )
        except RuntimeError:
            print("[nnUNetMD] find_best_configuration failed — run manually.")
    else:
        print(f"\n[nnUNetMD] WARNING: failed folds: {failed}\n"
              "  Fix and re-run with --skip-prepare.")

    print(
        f"\n[nnUNetMD] ✓ Stage 2 complete.\n"
        f"  Results: {res_base / new_ss_name}"
        f"/{args.trainer}__{args.plans_name}__3d_fullres"
    )
    return 0 if not failed else 1


def entry_point():
    sys.exit(main())


if __name__ == "__main__":
    entry_point()