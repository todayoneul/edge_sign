import argparse
import json
import os


def build_labels(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    labels = sorted(
        [
            name
            for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
        ]
    )
    if not labels:
        raise ValueError("No label directories found under the dataset path.")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Export label names to a JSON file for the web demo."
    )
    parser.add_argument(
        "--data-dir",
        default="./dataset/train",
        help="Path to the dataset train directory.",
    )
    parser.add_argument(
        "--output",
        default="./web/labels.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    labels = build_labels(args.data_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    main()
