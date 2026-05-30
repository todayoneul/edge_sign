import os
import urllib.request
import ssl
from pathlib import Path

# Disable SSL verification for stability in case of local cert errors
ssl._create_default_https_context = ssl._create_unverified_context

WEB_MODEL_DIR = Path(__file__).parent.parent / "web" / "model"

# Hugging Face repositories containing pre-converted MobileSAM ONNX files
MODELS = {
    "mobile_sam_image_encoder.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx",
    "mobile_sam_mask_decoder.onnx": "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx"
}

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = f"\rProgress: {percent:5.1f}% [{readsofar}/{totalsize} bytes]"
            print(s, end="")
        else:
            print(f"\rDownloaded {readsofar} bytes", end="")
    
    urllib.request.urlretrieve(url, dest, reporthook)
    print("\nDownload complete.")

def main():
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, url in MODELS.items():
        dest_path = WEB_MODEL_DIR / filename
        if dest_path.exists():
            print(f"{filename} already exists at {dest_path}. Skipping.")
            continue
            
        try:
            download_file(url, dest_path)
        except Exception as e:
            print(f"Failed to download {filename} from {url}. Error: {e}")

if __name__ == "__main__":
    main()
