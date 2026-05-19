import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter manifest to top-K frequent classes.")
    parser.add_argument("--manifest", default="./dataset/landmarks/manifest.csv", help="Input manifest CSV path.")
    parser.add_argument("--labels", default="./dataset/landmarks/labels.json", help="Input labels JSON path.")
    parser.add_argument("--topk", type=int, default=50, help="Number of classes to keep.")
    parser.add_argument("--out-dir", default="./dataset/landmarks_top50", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    counts = Counter()
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_id = int(row["label_id"])
            counts[label_id] += 1
            rows.append(row)

    if not counts:
        raise ValueError("Manifest is empty or invalid.")

    # Deterministic top-K: by frequency desc, then label text asc
    def sort_key(label_id):
        label_name = labels[label_id] if label_id < len(labels) else ""
        return (-counts[label_id], label_name)

    top_ids = sorted(counts.keys(), key=sort_key)[: args.topk]
    top_id_set = set(top_ids)

    new_labels = []
    old_to_new = {}
    for old_id in top_ids:
        old_to_new[old_id] = len(new_labels)
        label_name = labels[old_id] if old_id < len(labels) else f"label_{old_id}"
        new_labels.append(label_name)

    out_manifest = out_dir / "manifest.csv"
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label_id", "label"])
        for row in rows:
            old_id = int(row["label_id"])
            if old_id not in top_id_set:
                continue
            new_id = old_to_new[old_id]
            writer.writerow([row["path"], new_id, new_labels[new_id]])

    out_labels = out_dir / "labels.json"
    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, ensure_ascii=False, indent=2)

    print("[Success] Top-K filtering complete.")
    print(f"- Kept classes: {len(new_labels)}")
    print(f"- Output manifest: {out_manifest}")
    print(f"- Output labels: {out_labels}")


if __name__ == "__main__":
    main()
