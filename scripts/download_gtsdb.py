"""
GTSDB (German Traffic Sign Detection Benchmark) 다운로드 스크립트.
검출용 데이터셋: 이미지 + 바운딩박스 어노테이션 (CSV).

다운로드 후 data/GTSDB/ 에 저장됩니다.
  - TrainIJCNN2013/          : 학습 이미지 (00000.ppm ~ 00599.ppm, 600장)
  - TestIJCNN2013/           : 테스트 이미지 (00600.ppm ~ 00899.ppm, 300장)
  - gt.txt                   : 학습 어노테이션 (CSV: filename;x1;y1;x2;y2;classId)
  - ReadMe.txt               : 클래스 정의 (43개 세부 클래스 → 4개 상위 카테고리)
"""
import os
import zipfile
import urllib.request
import ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = Path(__file__).parent.parent / "data" / "GTSDB"

URLS = {
    "train_images": "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip",
}


def download_file(url, dest):
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            print(f"\r  Progress: {percent:5.1f}% [{readsofar:,}/{totalsize:,} bytes]", end="")
        else:
            print(f"\r  Downloaded {readsofar:,} bytes", end="")

    urllib.request.urlretrieve(url, dest, reporthook)
    print("\n  Download complete.")


def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("  Extraction complete.")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for key, url in URLS.items():
        zip_name = url.split("/")[-1]
        zip_path = DATA_DIR / zip_name

        if not zip_path.exists():
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue

        unzip_file(zip_path, DATA_DIR)

        try:
            zip_path.unlink()
            print(f"  Removed {zip_name}")
        except Exception as e:
            print(f"  Could not remove {zip_path}: {e}")

    print(f"\nGTSDB data saved to: {DATA_DIR}")
    train_dir = DATA_DIR / "FullIJCNN2013"
    if train_dir.exists():
        gt_file = train_dir / "gt.txt"
        ppm_files = list(train_dir.glob("*.ppm"))
        print(f"  Images: {len(ppm_files)}")
        if gt_file.exists():
            with open(gt_file, "r") as f:
                lines = f.readlines()
            print(f"  Annotations: {len(lines)} bounding boxes")


if __name__ == "__main__":
    main()
