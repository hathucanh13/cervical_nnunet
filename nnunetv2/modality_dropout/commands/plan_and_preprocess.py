"""
nnUNetMD_plan_and_preprocess
────────────────────────────
1. Run nnU-Net 3d_fullres planning on BOTH datasets (saves nnUNetPlans.json).
2. Load the auto-generated plans, patch the multimodal one with the
   single-sequence 3d_fullres block, save as nnUNetPlans_MD.json.
3. Fix normalization: ZScore for real channels, NoNormalization for zero-filled.
4. Optionally verify multimodal dataset integrity.
5. Preprocess the multimodal dataset using nnUNetPlans_MD.
6. Write nnUNetMD_metadata.json for downstream commands.
"""

import argparse
import sys

from nnunetv2.modality_dropout.utils import (
    count_channels,
    get_nnunet_base,
    load_plans,
    patch_multimodal_plans,
    resolve_dataset,
    run_cmd,
    save_plans,
    verify_dataset_integrity,
    write_metadata,
)

# nnU-Net's auto-generated plans name — never modified by this pipeline
_NNUNET_DEFAULT_PLANS = "nnUNetPlans"

# Our patched plans name — distinct to avoid collisions or accidental overwrites
_MD_PLANS_NAME = "nnUNetPlans_MD"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nnUNetMD_plan_and_preprocess",
        description=(
            "Plan and preprocess a multimodal + single-sequence dataset pair "
            "for nnU-Net modality-dropout training (3d_fullres only)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset_multimodal",
                   help="Multimodal dataset: numeric ID or full folder name.")
    p.add_argument("dataset_single_sequence",
                   help="Single-sequence dataset: numeric ID or full folder name.")
    p.add_argument(
        "--verify-dataset-integrity", action="store_true", default=False,
        help="Check image shapes in the multimodal dataset before preprocessing.",
    )
    p.add_argument(
        "-pl", default=_MD_PLANS_NAME,
        help=(
            f"Name for the patched plans file (default: {_MD_PLANS_NAME}). "
            "nnU-Net's original plans are always kept as nnUNetPlans.json — "
            "this name is for the modified version only."
        ),
    )
    p.add_argument("--num-processes", type=int, default=8)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    raw_base, pre_base, _ = get_nnunet_base()

    mm_id, mm_name, mm_raw = resolve_dataset(args.dataset_multimodal,      raw_base)
    ss_id, ss_name, ss_raw = resolve_dataset(args.dataset_single_sequence, raw_base)

    print(f"\n[nnUNetMD] Multimodal : {mm_name}  (ID={mm_id})")
    print(f"[nnUNetMD] Single-seq : {ss_name}  (ID={ss_id})")
    print( "[nnUNetMD] Config     : 3d_fullres\n")

    # ── Optional integrity check ──────────────────────────────────────────────
    if args.verify_dataset_integrity:
        if not verify_dataset_integrity(mm_raw):
            print("[nnUNetMD] Aborting — fix integrity errors first.")
            return 1

    # ── Step 1: Plan BOTH datasets ────────────────────────────────────────────
    # Always plan with nnU-Net's default name.
    # Do NOT pass --plans_name here — we read the output and save our own copy.
    for ds_id, ds_name in [(mm_id, mm_name), (ss_id, ss_name)]:
        print(f"\n[nnUNetMD] Planning {ds_name} …")
        run_cmd(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d", str(ds_id),
                "-c", "3d_fullres",
                "--no_pp",              # plan only — preprocessing comes later
            ],
            description=f"plan {ds_name}",
        )

    print(f"\n[nnUNetMD] Patching multimodal plans …")
    mm_plans = load_plans(pre_base, mm_name, _NNUNET_DEFAULT_PLANS)
    ss_plans = load_plans(pre_base, ss_name, _NNUNET_DEFAULT_PLANS)
 
    n_ch_mm = count_channels(mm_raw / "dataset.json")
    n_ch_ss = count_channels(ss_raw / "dataset.json")
 
    print(f"  Multimodal channels : {n_ch_mm}")
    print(f"  Single-seq channels : {n_ch_ss}")
 
    if n_ch_mm <= n_ch_ss:
        print(f"  [WARNING] Expected multimodal > single-seq channels "
              f"(got {n_ch_mm} vs {n_ch_ss}).")
 
    # All n_ch_mm channels in the multimodal dataset contain real image data
    # at this stage — ZScore applied to all, no zero-filled channels yet.
    patched = patch_multimodal_plans(
        mm_plans,
        ss_plans,
        n_channels_mm   = n_ch_mm,
        n_real_channels = n_ch_mm,
    )
 
    # ── Step 3: Save patched plans under overwrite_plans_name ─────────────────
    # Store channel counts so nnUNetTrainerStage2 can read them at runtime
    # and build the correct N-channel architecture without needing a _MD dataset.
    patched["nnUNetMD_n_channels_pretrained"] = n_ch_mm
    patched["nnUNetMD_n_channels_single"]     = n_ch_ss
 
    # nnU-Net's original plans (args.plans_name) are left untouched.
    save_plans(pre_base, mm_name, patched, args.pl)
 
    cfg = patched["configurations"]["3d_fullres"]
    print(f"  patch_size            : {cfg['patch_size']}")
    print(f"  batch_size            : {cfg['batch_size']}")
    print(f"  normalization_schemes : {cfg['normalization_schemes']}")
    print(f"  spacing               : {cfg['spacing']}")
    print(f"  Original plans        : {args.pl}.json              (untouched)")
    print(f"  Patched plans saved   : {args.pl}.json")

    # ── Step 4: Write metadata ────────────────────────────────────────────────
    write_metadata(
        preprocessed_base      = pre_base,
        multimodal_folder      = mm_name,
        single_seq_folder      = ss_name,
        n_channels_multimodal  = n_ch_mm,
        n_channels_single      = n_ch_ss,
    )

    print(f"\n[nnUNetMD] ✓ Done.\n"
          f"  Next: nnUNetMD_pretrain "
          f"{args.dataset_multimodal} {args.dataset_single_sequence}")
    return 0


def entry_point():
    sys.exit(main())


if __name__ == "__main__":
    entry_point()