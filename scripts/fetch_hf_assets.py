import argparse
import os
import urllib.request


def build_url(repo_id, revision, filename):
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def download(url, output_path, token=None, force=False):
    if os.path.exists(output_path) and not force:
        print(f"Skip (exists): {output_path}")
        return

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        data = response.read()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Download model assets from Hugging Face Hub."
    )
    parser.add_argument("--repo", required=True, help="Hugging Face repo id.")
    parser.add_argument("--revision", default="main", help="Repo revision.")
    parser.add_argument(
        "--model-file",
        default="convnextv2_ksl_int8.onnx",
        help="Model filename in the repo.",
    )
    parser.add_argument(
        "--labels-file",
        default="labels.json",
        help="Labels filename in the repo.",
    )
    parser.add_argument(
        "--output-dir",
        default="./web",
        help="Output base directory.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite files.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    model_url = build_url(args.repo, args.revision, args.model_file)
    labels_url = build_url(args.repo, args.revision, args.labels_file)

    download(
        model_url,
        os.path.join(args.output_dir, "model", args.model_file),
        token=token,
        force=args.force,
    )
    download(
        labels_url,
        os.path.join(args.output_dir, args.labels_file),
        token=token,
        force=args.force,
    )


if __name__ == "__main__":
    main()
