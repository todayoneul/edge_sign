import os
import zipfile
import urllib.request
import ssl
from pathlib import Path

# Disable SSL verification for download stability in case of local cert errors
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = Path(__file__).parent.parent / "data"
GTSRB_DIR = DATA_DIR / "GTSRB"

# Official download URLs as fallback
URLS = {
    "train": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1f12602/GTSRB_Final_Training_Images.zip",
    "test": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1f12602/GTSRB_Final_Test_Images.zip",
    "test_gt": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1f12602/GTSRB_Final_Test_GT.zip"
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

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try using torchvision first
    try:
        print("Attempting to download via torchvision.datasets.GTSRB...")
        import torchvision
        # This will download to data/gtsrb
        torchvision.datasets.GTSRB(root=str(DATA_DIR), split="train", download=True)
        torchvision.datasets.GTSRB(root=str(DATA_DIR), split="test", download=True)
        print("Successfully downloaded via torchvision.")
        return
    except Exception as e:
        print(f"torchvision download failed or not installed. Error: {e}")
        print("Falling back to direct HTTP downloads from official repository...")

    # Fallback to direct download
    for key, url in URLS.items():
        zip_name = url.split("/")[-1]
        zip_path = DATA_DIR / zip_name
        
        # Check if already extracted or zip exists
        if not zip_path.exists():
            try:
                download_file(url, zip_path)
            except Exception as dl_err:
                print(f"Failed to download {url}. Error: {dl_err}")
                continue
                
        # Unzip
        unzip_file(zip_path, DATA_DIR)
        
        # Clean up zip to save space
        try:
            zip_path.unlink()
            print(f"Removed temporary archive {zip_name}")
        except Exception as rm_err:
            print(f"Could not remove {zip_path}: {rm_err}")

if __name__ == "__main__":
    main()
