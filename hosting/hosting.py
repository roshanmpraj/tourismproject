from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "Roshanmpraj/tourismproject"  # Your Hugging Face Space repo
repo_type = "space"

# Local folder to upload (adjust path if needed)
folder_path = "tourism_project/deployment"

# Check if the folder exists
if not os.path.isdir(folder_path):
    raise ValueError(f"Provided path: '{folder_path}' does not exist or is not a directory")

# Step 1: Check if the Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new Space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload deployment folder
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""  # Upload to root of the repo
)

print("✅ Files uploaded successfully to Hugging Face Space!")
