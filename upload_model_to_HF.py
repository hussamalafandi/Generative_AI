import os
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

def main():
    # Load Hugging Face credentials from .env
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")

    if not hf_token or not hf_username:
        print("HF_TOKEN or HF_USERNAME not found in .env")
        return

    # Get model folder and repo name from user
    model_path = input("Enter the path to your model folder: ").strip()
    if not os.path.isdir(model_path):
        print("Provided path is not a directory.")
        return

    repo_name = input("Enter the Hugging Face repo name you want to create (e.g., 'my-cool-model'): ").strip()
    repo_id = f"{hf_username}/{repo_name}"

    # Create repo (if it doesn't exist)
    try:
        create_repo(repo_id=repo_id, token=hf_token)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Failed to create repo: {e}")
        return

    # Upload model folder
    try:
        print(f"Uploading files from '{model_path}' to '{repo_id}'...")
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=hf_token,
            path_in_repo=".",  # Upload to the root of the repo
            ignore_patterns=["*.tmp", "*.log", "__pycache__"],
        )
        print(f"Upload complete! View it at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
