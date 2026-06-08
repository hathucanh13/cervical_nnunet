"""
nnUNetMD_train_from_pretrain  —  entry point
─────────────────────────────────────────────
Stage 2: load pretrained multimodal weights and run standard nnU-Net
cross-validation on a zero-filled version of the single-sequence dataset.

Instead of padding channels on the fly, we first materialise a new raw
dataset where every auxiliary channel (beyond T2w / ch0) is written to
disk as a zero-filled NIfTI that matches the anchor channel's geometry.
This keeps the nnU-Net data pipeline completely standard — no custom
trainer padding logic is needed.

Steps
─────
  1. Verify pretrain checkpoint exists (from nnUNetMD_metadata.json).
  2. Create the zero-filled dataset on disk (Dataset{N}_ZF_{ss_name}/)
     unless --skip-create-zf is given.
  3. Preprocess the new zero-filled dataset with Stage-1 plans
     (correct geometry: patch_size, spacing) and channel-aware normalization:
       ch0        : ZScoreNormalization  (real T2w data)
       ch1…chN-1  : NoNormalization      (zero-filled — must stay at 0.0)
  4. Run cross-validation on the zero-filled dataset, loading pretrained
     weights for each fold.
"""

import argparse
import sys
from pathlib import Path

from nnunetv2.modality_dropout.utils import (
    count_channels,
    create_zerofilled_dataset,
    get_nnunet_base,
    load_metadata,
    load_plans,
    patch_multimodal_plans,
    resolve_dataset,
    run_cmd,
    save_plans,
    update_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nnUNetMD_train_from_pretrain",
        description=(
            "Stage 2: load pretrained multimodal weights and run standard\n"
            "nnU-Net cross-validation on a zero-filled single-sequence dataset.\n\n"
            "A new raw dataset (Dataset{N}_ZF_{ss_name}) is created on disk\n"
            "where every auxiliary channel is a zero-filled image matching\n"
            "the T2w anchor geometry.  No on-the-fly padding is applied.\n\n"
            "Run 'nnUNetMD_pretrain' first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset_multimodal",
                   help="Multimodal dataset: numeric ID or full folder name.")
    p.add_argument("dataset_single_sequence",
                   help="Single-sequence dataset: numeric ID or full folder name.")
    p.add_argument(
        "--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Folds to train (default: 0 1 2 3 4).",
    )
    p.add_argument(
        "--trainer", default="nnUNetTrainerStage2",
        help="Trainer class for Stage 2 (default: nnUNetTrainerStage2).",
    )
    p.add_argument(
        "-plans_name", dest="plans_name", default="nnUNetPlans_MD",
        help=(
            "Plans name to use for Stage-2 preprocessing and training. "
            "Must be the patched plans produced by nnUNetMD_plan_and_preprocess "
            "(default: nnUNetPlans_MD)."
        ),
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Override pretrained checkpoint path (default: read from metadata).",
    )
    p.add_argument(
        "--num-processes", type=int, default=8,
        help="Parallel processes for preprocessing (default: 8).",
    )
    p.add_argument(
        "--skip-create-zf", action="store_true", default=False,
        help=(
            "Skip zero-filled dataset creation if it already exists "
            "(e.g. when resuming a failed run)."
        ),
    )
    p.add_argument(
        "--skip-preprocess", action="store_true", default=False,
        help="Skip preprocessing if already done in a previous run.",
    )
    p.add_argument(
        "--zf-dataset-id", type=int, default=None,
        help=(
            "Override the numeric ID assigned to the new zero-filled dataset "
            "(default: next available ID in nnUNet_raw)."
        ),
    )
    return p


def _prepare_zf_plans(pre_base: Path, mm_name: str, zf_name: str,
                      n_ch_mm: int, n_ch_ss: int, plans_name: str) -> None:
    """
    Create Stage-2 plans for the zero-filled dataset by patching Stage-1
    multimodal plans:
      ch0               : ZScoreNormalization  (real T2w data)
      ch1 … ch(N-1)     : NoNormalization      (zero-filled — must stay 0.0)

    Geometry (patch_size, spacing) comes from the multimodal Stage-1 plans,
    ensuring the preprocessed data is compatible with the pretrained weights.
    """
    mm_plans = load_plans(pre_base, mm_name, plans_name)
    # Use the mm plans as both source of geometry AND as the base to patch;
    # we only need ZF-specific normalization overrides.
    # The ZF dataset has n_ch_mm channels total, n_ch_ss of which are real.
    import copy
    zf_plans = copy.deepcopy(mm_plans)

    cfg = zf_plans["configurations"]["3d_fullres"]
    base_scheme = cfg.get("normalization_schemes",
                          ["ZScoreNormalization"] * n_ch_mm)[0]

    cfg["normalization_schemes"] = (
        [base_scheme]         * n_ch_ss
        + ["NoNormalization"] * (n_ch_mm - n_ch_ss)
    )
    cfg["use_mask_for_norm"] = [False] * n_ch_mm

    zf_plans["dataset_name"] = zf_name
    # Carry over nnUNetMD channel metadata so nnUNetTrainerStage2 reads them
    zf_plans["nnUNetMD_n_channels_pretrained"] = n_ch_mm
    zf_plans["nnUNetMD_n_channels_single"]     = n_ch_ss

    (pre_base / zf_name).mkdir(parents=True, exist_ok=True)
    save_plans(pre_base, zf_name, zf_plans, plans_name)

    print(f"  patch_size            : {cfg['patch_size']}")
    print(f"  normalization_schemes : {cfg['normalization_schemes']}")
    print(f"  n_channels_pretrained : {n_ch_mm}  (all present on disk)")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raw_base, pre_base, res_base = get_nnunet_base()

    mm_id, mm_name, mm_raw = resolve_dataset(args.dataset_multimodal,      raw_base)
    ss_id, ss_name, ss_raw = resolve_dataset(args.dataset_single_sequence, raw_base)

    # ── Verify pretrain complete ───────────────────────────────────────────────
    meta = load_metadata(pre_base, mm_name)
    if not meta.get("pretrain_complete", False):
        print(f"[nnUNetMD] ERROR: pretrain not complete.\n"
              f"  Run: nnUNetMD_pretrain {args.dataset_multimodal} "
              f"{args.dataset_single_sequence}")
        return 1

    checkpoint = args.checkpoint or meta.get("pretrain_checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        print(f"[nnUNetMD] ERROR: checkpoint not found: {checkpoint}")
        return 1

    n_ch_mm = meta["n_channels_multimodal"]
    n_ch_ss = meta["n_channels_single"]

    # ── Step 1: Create zero-filled dataset on disk ────────────────────────────
    if not args.skip_create_zf:
        zf_id, zf_name, zf_path = create_zerofilled_dataset(
            ss_raw   = ss_raw,
            mm_raw   = mm_raw,
            raw_base = raw_base,
            new_id   = args.zf_dataset_id,
        )
    else:
        # Retrieve from metadata (written by a previous run)
        zf_name = meta.get("zerofilled_dataset")
        zf_id   = meta.get("zerofilled_dataset_id")
        if not zf_name:
            print("[nnUNetMD] ERROR: --skip-create-zf used but no zero-filled "
                  "dataset recorded in metadata.  Remove the flag for the first run.")
            return 1
        zf_path = raw_base / zf_name
        if not zf_path.exists():
            print(f"[nnUNetMD] ERROR: zero-filled dataset not found: {zf_path}")
            return 1
        print(f"[nnUNetMD] Using existing zero-filled dataset: {zf_name}")

    # Persist ZF dataset info in metadata so downstream tools can find it
    update_metadata(pre_base, mm_name,
                    zerofilled_dataset    = zf_name,
                    zerofilled_dataset_id = zf_id)

    print(f"\n[nnUNetMD] Stage 2 fine-tuning  (disk-based zero-filled dataset)")
    print(f"  Zero-filled dataset : {zf_name}  (ID={zf_id})")
    print(f"  Checkpoint          : {checkpoint}")
    print(f"  Trainer             : {args.trainer}")
    print(f"  Folds               : {args.folds}")
    print(f"  Channel layout      : ch0=T2w (real), ch1–{n_ch_mm-1}=zeros on disk")

    # ── Step 2: Prepare Stage-2 plans for zero-filled dataset ─────────────────
    if not args.skip_preprocess:
        print(f"\n[nnUNetMD] Preparing Stage-2 plans for {zf_name} …")
        _prepare_zf_plans(pre_base, mm_name, zf_name,
                          n_ch_mm, n_ch_ss, args.plans_name)

        # Preprocess the zero-filled dataset with Stage-1 geometry
        print(f"\n[nnUNetMD] Preprocessing {zf_name} with {args.plans_name} …")
        run_cmd(
            [
                "nnUNetv2_preprocess",
                "-d", str(zf_id),
                "-c", "3d_fullres",
                "-plans_name", args.plans_name,
                "-np", str(args.num_processes),
            ],
            description=f"preprocess {zf_name}",
        )
    else:
        print("  [skip] Preprocessing skipped.")

    # ── Step 3: Cross-validation ──────────────────────────────────────────────
    print(f"\n[nnUNetMD] Cross-validation on {zf_name}")
    failed = []
    for fold in args.folds:
        print(f"\n[nnUNetMD] ── fold {fold} ──")
        try:
            run_cmd(
                [
                    "nnUNetv2_train",
                    str(zf_id),
                    "3d_fullres",
                    str(fold),
                    "-tr",    args.trainer,
                    "-p",     args.plans_name,
                    "-pretrained_weights", checkpoint,
                    "--npz",
                ],
                f"Stage 2 fold {fold}",
            )
        except RuntimeError as e:
            print(f"[nnUNetMD] WARNING fold {fold} failed: {e}")
            failed.append(fold)

    # ── Step 4: Find best configuration ───────────────────────────────────────
    if not failed:
        print("\n[nnUNetMD] Finding best configuration …")
        try:
            run_cmd(
                [
                    "nnUNetv2_find_best_configuration",
                    str(zf_id),
                    "-c", "3d_fullres",
                    "-tr", args.trainer,
                    "-p",  args.plans_name,
                ],
                "find_best_configuration",
            )
        except RuntimeError:
            print("[nnUNetMD] find_best_configuration failed — run manually.")
    else:
        print(f"\n[nnUNetMD] WARNING: failed folds: {failed}\n"
              "  Re-run with --skip-create-zf --skip-preprocess to skip "
              "data preparation.")

    print(
        f"\n[nnUNetMD] ✓ Stage 2 complete.\n"
        f"  Results: {res_base / zf_name}"
        f"/{args.trainer}__{args.plans_name}__3d_fullres"
    )
    return 0 if not failed else 1


def entry_point():
    sys.exit(main())


if __name__ == "__main__":
    entry_point()