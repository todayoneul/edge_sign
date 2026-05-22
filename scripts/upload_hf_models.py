import os
import argparse
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload packaged KSL models to Hugging Face Hub.")
    parser.add_argument("--token", help="Hugging Face Write Token. If not provided, it will read from .env.")
    parser.add_argument("--repo-prefix", default="edge-sign-ksl", help="Prefix for the repositories (e.g. edge-sign-ksl).")
    args = parser.parse_args()
    
    # 1. Resolve token
    token = args.token
    if not token:
        # Try loading from .env
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "hf_readtoken=" in line:
                        # Extract token
                        parts = line.strip().split("=")
                        if len(parts) == 2:
                            token = parts[1].strip()
                            print("Found token in .env file.")
                            break
                            
    if not token:
        print("Error: Hugging Face Write Token이 없습니다.")
        print("Token은 https://huggingface.co/settings/tokens 에서 발급받을 수 있습니다. (Role: Write)")
        print("사용법: python scripts/upload_hf_models.py --token YOUR_WRITE_TOKEN")
        return
        
    print("Warning: .env에 저장된 토큰이 Read 전용일 경우 업로드가 실패할 수 있습니다. 실패 시 Write 권한이 있는 토큰을 사용해주세요.")
    
    # Initialize HfApi
    api = HfApi()
    
    try:
        # Get username from token
        user_info = api.whoami(token=token)
        username = user_info["name"]
        print(f"Authenticated successfully as user: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return
        
    # Model repositories to upload
    models_to_upload = {
        "mediapipe": {
            "local_dir": "./models/hf_mediapipe_ksl",
            "repo_name": f"{args.repo_prefix}-mediapipe"
        },
        "landmark": {
            "local_dir": "./models/hf_landmark_ksl",
            "repo_name": f"{args.repo_prefix}-landmark"
        }
    }
    
    for key, info in models_to_upload.items():
        local_dir = info["local_dir"]
        repo_id = f"{username}/{info['repo_name']}"
        
        if not os.path.exists(local_dir):
            print(f"\n[Error] {local_dir} 디렉토리가 없습니다. 먼저 package_hf_models.py를 실행하세요.")
            continue
            
        print(f"\n--- Uploading {key.upper()} Model to repo: {repo_id} ---")
        
        # Create repository if not exists
        try:
            create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
            print(f"Repository {repo_id} ready (created or already exists).")
        except Exception as e:
            print(f"Failed to create/verify repository {repo_id}: {e}")
            continue
            
        # Upload folder
        try:
            print(f"Uploading files from {local_dir} to {repo_id}...")
            api.upload_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"Successfully uploaded {key.upper()} Model to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Failed to upload folder {local_dir} to {repo_id}: {e}")
            
    print("\nUpload process finished!")

if __name__ == "__main__":
    main()
