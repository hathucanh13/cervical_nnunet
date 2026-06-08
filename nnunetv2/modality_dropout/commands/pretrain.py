"""
nnUNetMD_pretrain  —  entry point
"""

import argparse
import re
import sys
from pathlib import Path

from nnunetv2.modality_dropout.utils import (
    get_nnunet_base,
    load_metadata,
    resolve_dataset,
    run_cmd,
    update_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nnUNetMD_pretrain",
        description=(
            "Stage 1: train the modality-dropout model on ALL cases "
            "of the multimodal dataset (fold=all).\n"
            "Run 'nnUNetMD_plan_and_preprocess' first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset_multimodal")
    p.add_argument("dataset_single_sequence")
    p.add_argument("--trainer", default="nnUNetTrainerModalityDropout")
    p.add_argument("--plans-name", default="nnUNetPlans")
    p.add_argument(
        "--max-epochs", type=int, default=None,
        help=(
            "Training epochs. If omitted, reads best epoch from a completed "
            "fold_0 log; otherwise defaults to nnU-Net default (1000)."
        ),
    )
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    return p


def _best_epoch_from_fold0(res_base: Path, mm_name: str,
                            trainer: str, plans_name: str) -> int | None:
    """Parse training_log.txt of fold_0 to find the epoch with peak val Dice."""
    log = (res_base / mm_name
           / f"{trainer}__{plans_name}__3d_fullres"
           / "fold_0" / "training_log.txt")
    if not log.exists():
        return None

    best_epoch, best_dice = None, -1.0
    pattern = re.compile(r"Epoch\s+(\d+).*?Pseudo dice\s*\[([0-9.]+)\]", re.S)
    for m in pattern.finditer(log.read_text(errors="replace")):
        epoch, dice = int(m.group(1)), float(m.group(2))
        if dice > best_dice:
            best_dice, best_epoch = dice, epoch

    if best_epoch is not None:
        print(f"[nnUNetMD] fold_0 best epoch: {best_epoch}  "
              f"(val Dice = {best_dice:.4f})")
    return best_epoch


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raw_base, pre_base, res_base = get_nnunet_base()

    mm_id, mm_name, _ = resolve_dataset(args.dataset_multimodal,      raw_base)
    _,      ss_name, _ = resolve_dataset(args.dataset_single_sequence, raw_base)

    # Verify prerequisites
    meta = load_metadata(pre_base, mm_name)
    if meta["single_sequence_dataset"] != ss_name:
        print(f"[nnUNetMD] WARNING: metadata records single-seq as "
              f"'{meta['single_sequence_dataset']}' but '{ss_name}' was passed.")

    # Determine max epochs
    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = _best_epoch_from_fold0(
            res_base, mm_name, args.trainer, args.plans_name)
    if max_epochs is None:
        print("[nnUNetMD] No fold_0 log found — using nnU-Net default (1000 epochs).")
        max_epochs = 1000

    print(f"\n[nnUNetMD] Stage 1 pretraining")
    print(f"  Dataset    : {mm_name}  (ID={mm_id})")
    print(f"  Trainer    : {args.trainer}")
    print(f"  Fold       : all")
    print(f"  Max epochs : {max_epochs}")

    # Build train command
    if args.num_gpus > 1:
        cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}",
               "-m", "nnunetv2.run.run_training",
               str(mm_id), "3d_fullres", "all",
               "-tr", args.trainer, "-p", args.plans_name, "--npz"]
    else:
        cmd = ["nnUNetv2_train", str(mm_id), "3d_fullres", "all",
               "-tr", args.trainer, "-p", args.plans_name, "--npz"]

    run_cmd(cmd, "Stage 1 fold_all")

    # Locate checkpoint (fold_all has no val split → only final is guaranteed)
    ckpt_dir   = (res_base / mm_name
                  / f"{args.trainer}__{args.plans_name}__3d_fullres"
                  / "fold_all")
    ckpt_best  = ckpt_dir / "checkpoint_best.pth"
    ckpt_final = ckpt_dir / "checkpoint_final.pth"
    chosen     = ckpt_best if ckpt_best.exists() else ckpt_final

    if not chosen.exists():
        print(f"[nnUNetMD] WARNING: checkpoint not found at {chosen}")
        return 1

    update_metadata(pre_base, mm_name,
                    pretrain_checkpoint=str(chosen),
                    pretrain_complete=True)

    print(f"\n[nnUNetMD] ✓ Pretraining complete.\n"
          f"  Checkpoint : {chosen}\n"
          f"  Next: nnUNetMD_train_from_pretrain "
          f"{args.dataset_multimodal} {args.dataset_single_sequence}")
    return 0


def entry_point():
    sys.exit(main())


if __name__ == "__main__":
    entry_point()