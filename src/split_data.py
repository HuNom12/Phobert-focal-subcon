"""CLI and script for preparing and splitting datasets."""
import argparse
from src.data import prepare_real_dataset, create_dummy_data


def main():
    parser = argparse.ArgumentParser(description="Chia và tiền xử lý tập dữ liệu.")
    parser.add_argument("--dataset", type=str, default="victsd", choices=["uit-vsfc", "victsd", "dummy"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "dummy":
        out_dir = args.output_dir or "data/dummy"
        create_dummy_data(output_dir=out_dir)
    else:
        out_dir = args.output_dir or "data/processed"
        prepare_real_dataset(args.dataset, output_dir=out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
