import argparse
import json
import sys
from pathlib import Path

# Ensure local imports work when running as a script.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    evaluate_group_kfold,
    evaluate_leave_one_out,
    evaluate_time_split,
    reconstruct,
    train_model,
)


def _load_config(path: Path):
    if not path:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_list(value: str):
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="SUEZ 2 baseline pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train baseline model")
    train_parser.add_argument("--data-dir", required=True, type=Path)
    train_parser.add_argument("--model-dir", required=True, type=Path)
    train_parser.add_argument("--config", type=Path, default=None)
    train_parser.add_argument("--holdout", type=str, default=None)
    train_parser.add_argument("--max-rows", type=int, default=None)

    recon_parser = subparsers.add_parser("reconstruct", help="Generate predictions")
    recon_parser.add_argument("--data-dir", required=True, type=Path)
    recon_parser.add_argument("--model-dir", required=True, type=Path)
    recon_parser.add_argument("--out-dir", required=True, type=Path)
    recon_parser.add_argument("--bss-ids", type=str, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Leave-one-out evaluation")
    eval_parser.add_argument("--data-dir", required=True, type=Path)
    eval_parser.add_argument("--out-dir", required=True, type=Path)
    eval_parser.add_argument("--mode", type=str, default="static")
    eval_parser.add_argument("--strategy", type=str, default="groupkfold")
    eval_parser.add_argument("--splits", type=int, default=5)
    eval_parser.add_argument("--max-items", type=int, default=None)
    eval_parser.add_argument("--test-ratio", type=float, default=0.2)
    eval_parser.add_argument("--min-obs", type=int, default=365)

    args = parser.parse_args()

    if args.command == "train":
        config = _load_config(args.config) or DEFAULT_CONFIG
        if args.holdout:
            config = {**config, "holdout_ids": _parse_list(args.holdout)}
        if args.max_rows:
            config = {**config, "max_rows": args.max_rows}
        result = train_model(args.data_dir, args.model_dir, config=config)
        print(json.dumps(result, indent=2))
    elif args.command == "reconstruct":
        bss_ids = _parse_list(args.bss_ids)
        result = reconstruct(args.data_dir, args.model_dir, args.out_dir, bss_ids=bss_ids)
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        if args.strategy == "loo":
            result = evaluate_leave_one_out(
                args.data_dir,
                args.out_dir,
                feature_mode=args.mode,
                max_items=args.max_items,
            )
        elif args.strategy == "timesplit":
            result = evaluate_time_split(
                args.data_dir,
                args.out_dir,
                feature_mode=args.mode,
                test_ratio=args.test_ratio,
                min_obs=args.min_obs,
            )
        else:
            result = evaluate_group_kfold(
                args.data_dir,
                args.out_dir,
                feature_mode=args.mode,
                n_splits=args.splits,
            )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
