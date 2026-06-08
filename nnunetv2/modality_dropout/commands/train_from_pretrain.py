"""
nnUNetMD_train_from_pretrain  —  entry point
─────────────────────────────────────────────
Stage 2: load pretrained multimodal weights and run standard nnU-Net
cross-validation on the ORIGINAL single-sequence dataset.

Zero-filling of auxiliary channels is handled on the fly by
nnUNetTrainerStage2 — no _MD dataset is created on disk.

Steps:
  1. Verify pretrain checkpoint exists (from nnUNetMD_metadata.json).
  2. Preprocess the single-sequence dataset with Stage-1 plans
     (correct geometry: patch_size, spacing) but single-channel normalization.
  3. Run cross-validation, loading pretrained weights for each fold.
"""

import argparse
import sys
from pathlib import Path

from nnunetv2.modality_dropout.utils import (
    count_channels,
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
            "Stage 2: load pretrained multimodal weights and run standard "
            "nnU-Net cross-validation on the single-sequence dataset.\n"
            "Zero-padding of auxiliary channels is applied on the fly — "
            "no _MD dataset is created on disk.\n\n"
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
        "--skip-preprocess", action="store_true", default=False,
        help="Skip preprocessing if already done in a previous run.",
    )
    return p


def _prepare_ss_plans(pre_base: Path, mm_name: str, ss_name: str,
                       n_ch_mm: int, n_ch_ss: int, plans_name: str) -> None:
    """
    Create Stage-2 plans for the single-sequence dataset by patching
    Stage-1 plans with n_real_channels=n_ch_ss:
      ch0        : ZScoreNormalization  (real data)
      ch1…chN-1  : NoNormalization      (will be zero-padded on the fly)

    Geometry (patch_size, spacing) comes from Stage-1 plans so the
    preprocessed data is compatible with the pretrained architecture.
    """
    mm_plans = load_plans(pre_base, mm_name, plans_name)
    ss_plans = load_plans(pre_base, ss_name, "nnUNetPlans")   # ss own default plans

    patched = patch_multimodal_plans(
        mm_plans,
        ss_plans,
        n_channels_mm   = n_ch_ss,      # ss dataset only has n_ch_ss channels
        n_real_channels = n_ch_ss,      # all of them are real
    )
    patched["dataset_name"] = ss_name

    # Carry over nnUNetMD channel metadata so the trainer can read them
    patched["nnUNetMD_n_channels_pretrained"] = n_ch_mm
    patched["nnUNetMD_n_channels_single"]     = n_ch_ss

    (pre_base / ss_name).mkdir(parents=True, exist_ok=True)
    save_plans(pre_base, ss_name, patched, plans_name)

    cfg = patched["configurations"]["3d_fullres"]
    print(f"  patch_size            : {cfg['patch_size']}")
    print(f"  normalization_schemes : {cfg['normalization_schemes']}")
    print(f"  n_channels_pretrained : {n_ch_mm}  (padding applied on the fly)")


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

    print(f"\n[nnUNetMD] Stage 2 fine-tuning  (on-the-fly zero-padding)")
    print(f"  Single-seq dataset : {ss_name}  (ID={ss_id})")
    print(f"  Checkpoint         : {checkpoint}")
    print(f"  Trainer            : {args.trainer}")
    print(f"  Folds              : {args.folds}")
    print(f"  Channel padding    : {n_ch_ss} → {n_ch_mm}  (at batch load time)")

    # ── Step 1: Prepare Stage-2 plans for single-sequence dataset ─────────────
    if not args.skip_preprocess:
        print(f"\n[nnUNetMD] Preparing Stage-2 plans for {ss_name} …")
        _prepare_ss_plans(pre_base, mm_name, ss_name,
                          n_ch_mm, n_ch_ss, args.plans_name)

        # Preprocess single-sequence dataset with Stage-1 geometry plans
        # (-plans_name points to our patched file with the right patch_size/spacing)
        print(f"\n[nnUNetMD] Preprocessing {ss_name} with {args.plans_name} …")
        run_cmd(
            [
                "nnUNetv2_preprocess",
                "-d", str(ss_id),
                "-c", "3d_fullres",
                "-plans_name", args.plans_name,
                "-np", str(args.num_processes),
            ],
            description=f"preprocess {ss_name}",
        )
    else:
        print(f"  [skip] Preprocessing skipped.")

    # ── Step 2: Cross-validation ──────────────────────────────────────────────
    print(f"\n[nnUNetMD] Cross-validation on {ss_name}")
    failed = []
    for fold in args.folds:
        print(f"\n[nnUNetMD] ── fold {fold} ──")
        try:
            run_cmd(
                [
                    "nnUNetv2_train",
                    str(ss_id),
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

    # ── Step 3: Find best configuration ──────────────────────────────────────
    if not failed:
        print("\n[nnUNetMD] Finding best configuration …")
        try:
            run_cmd(
                [
                    "nnUNetv2_find_best_configuration",
                    str(ss_id),
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
              "  Re-run with --skip-preprocess to skip data preparation.")

    print(
        f"\n[nnUNetMD] ✓ Stage 2 complete.\n"
        f"  Results: {res_base / ss_name}"
        f"/{args.trainer}__{args.plans_name}__3d_fullres"
    )
    return 0 if not failed else 1


def entry_point():
    sys.exit(main())


if __name__ == "__main__":
    entry_point()